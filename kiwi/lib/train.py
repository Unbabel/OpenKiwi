#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2019 Unbabel <openkiwi@unbabel.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import logging
from pathlib import Path
from pprint import pformat

import torch

from kiwi import constants as const
from kiwi.cli.pipelines.train import build_parser
from kiwi.data import builders, utils
from kiwi.data.iterators import build_bucket_iterator
from kiwi.data.utils import (
    save_training_datasets,
    save_vocabularies_from_datasets,
)
from kiwi.lib.utils import (
    configure_logging,
    configure_seed,
    merge_namespaces,
    save_args_to_file,
    setup_output_directory,
)
from kiwi.loggers import tracking_logger
from kiwi.models.linear_word_qe_classifier import LinearWordQEClassifier
from kiwi.models.model import Model
from kiwi.trainers.callbacks import Checkpoint
from kiwi.trainers.linear_word_qe_trainer import LinearWordQETrainer
from kiwi.trainers.trainer import Trainer
from kiwi.trainers.utils import optimizer_class

logger = logging.getLogger(__name__)


class TrainRunInfo:
    """
    Encapsulates relevant information on training runs.

    Can be instantiated with a trainer object.

    Attributes:
        stats: Stats of the best model so far
        model_path: Path of the best model so far
        run_uuid: Unique identifier of the current run
    """

    def __init__(self, trainer):
        # FIXME: linear trainer not yet supported here
        #   (no full support to checkpointer)
        self.stats = trainer.checkpointer.best_stats()
        self.model_path = trainer.checkpointer.best_model_path()
        self.run_uuid = tracking_logger.run_uuid


def train_from_file(filename):
    """
    Loads options from a config file and calls the training procedure.

    Args:
        filename (str): filename of the configuration file
    """
    parser = build_parser()
    options = parser.parse_config_file(filename)
    return train_from_options(options)


def train_from_options(options):
    """
    Runs the entire training pipeline using the configuration options received.

    These options include the pipeline and model options plus the model's API.

    Args:
        options (Namespace): All the configuration options retrieved
            from either a config file or input flags and the model
            being used.
    """
    if options is None:
        return

    pipeline_options = options.pipeline
    model_options = options.model
    ModelClass = options.model_api

    tracking_run = tracking_logger.configure(
        run_uuid=pipeline_options.run_uuid,
        experiment_name=pipeline_options.experiment_name,
        run_name=pipeline_options.run_name,
        tracking_uri=pipeline_options.mlflow_tracking_uri,
        always_log_artifacts=pipeline_options.mlflow_always_log_artifacts,
    )

    with tracking_run:
        output_dir = setup(
            output_dir=pipeline_options.output_dir,
            seed=pipeline_options.seed,
            gpu_id=pipeline_options.gpu_id,
            debug=pipeline_options.debug,
            quiet=pipeline_options.quiet,
        )

        all_options = merge_namespaces(pipeline_options, model_options)
        log(
            output_dir,
            config_options=vars(all_options),
            save_config=pipeline_options.save_config,
        )

        trainer = run(ModelClass, output_dir, pipeline_options, model_options)
        train_info = TrainRunInfo(trainer)

    teardown(pipeline_options)

    return train_info


def run(ModelClass, output_dir, pipeline_options, model_options):
    """
    Implements the main logic of the training module.

    Instantiates the dataset, model class and sets their attributes according
    to the pipeline options received. Loads or creates a trainer and runs it.

    Args:
        ModelClass (Model): Python Type of the Model to train
        output_dir: Directory to save models
        pipeline_options (Namespace): Generic Train Options
            load_model: load pre-trained predictor model
            resume: load trainer state and resume training
            gpu_id: Set to non-negative integer to train on GPU
            train_batch_size: Batch Size for training
            valid_batch_size: Batch size for validation

        model_options(Namespace): Model Specific options

    Returns:
        The trainer object
    """
    model_name = getattr(ModelClass, "title", ModelClass.__name__)
    logger.info("Training the {} model".format(model_name))
    # FIXME: make sure all places use output_dir
    # del pipeline_options.output_dir
    pipeline_options.output_dir = None

    # Data step
    fieldset = ModelClass.fieldset(
        wmt18_format=model_options.__dict__.get("wmt18_format")
    )

    datasets = retrieve_datasets(
        fieldset, pipeline_options, model_options, output_dir
    )
    save_vocabularies_from_datasets(output_dir, *datasets)
    if pipeline_options.save_data:
        save_training_datasets(pipeline_options.save_data, *datasets)

    # Trainer step
    device_id = None
    if pipeline_options.gpu_id is not None and pipeline_options.gpu_id >= 0:
        device_id = pipeline_options.gpu_id

    vocabs = utils.fields_to_vocabs(datasets[0].fields)

    trainer = retrieve_trainer(
        ModelClass,
        pipeline_options,
        model_options,
        vocabs,
        output_dir,
        device_id,
    )

    logger.info(str(trainer.model))
    logger.info("{} parameters".format(trainer.model.num_parameters()))
    tracking_logger.log_param(
        "model_parameters", trainer.model.num_parameters()
    )

    # Dataset iterators
    train_iter = build_bucket_iterator(
        datasets[0],
        batch_size=pipeline_options.train_batch_size,
        is_train=True,
        device=device_id,
    )
    valid_iter = build_bucket_iterator(
        datasets[1],
        batch_size=pipeline_options.valid_batch_size,
        is_train=False,
        device=device_id,
    )

    trainer.run(train_iter, valid_iter, epochs=pipeline_options.epochs)

    return trainer


def retrieve_trainer(
    ModelClass, pipeline_options, model_options, vocabs, output_dir, device_id
):
    """
    Creates a Trainer object with an associated model.

    This object encapsulates the logic behind training the model and
    checkpointing. This method uses the received pipeline options to
    instantiate a Trainer object with the the requested model and
    hyperparameters.

    Args:
        ModelClass
        pipeline_options (Namespace): Generic training options
            resume (bool): Set to true if resuming an existing run.
            load_model (str): Directory containing model.torch for loading
                pre-created model.
            checkpoint_save (bool): Boolean indicating if snapshots should be
                saved after validation runs. warning: if false, will never save
                the model.
            checkpoint_keep_only_best (int): Indicates kiwi to keep the best
                `n` models.
            checkpoint_early_stop_patience (int): Stops training if metrics
                don't improve after `n` validation runs.
            checkpoint_validation_steps (int): Perform validation every `n`
                training steps.
            optimizer (string): The optimizer to be used in training.
            learning_rate (float): Starting learning rate.
            learning_rate_decay (float): Factor of learning rate decay.
            learning_rate_decay_start (int): Start decay after epoch `x`.
            log_interval (int): Log after `k` batches.
        model_options (Namespace): Model specific options.
        vocabs (dict): Vocab dictionary.
        output_dir (str or Path): Output directory for models and stats
            concerning training.
        device_id (int): The gpu id to be used in training. Set to negative
            to use cpu.
    Returns:
        Trainer

    """

    if pipeline_options.resume:
        return Trainer.resume(local_path=output_dir, device_id=device_id)

    if pipeline_options.load_model:
        model = Model.create_from_file(pipeline_options.load_model)
    else:
        model = ModelClass.from_options(vocabs=vocabs, opts=model_options)

    checkpointer = Checkpoint(
        output_dir,
        pipeline_options.checkpoint_save,
        pipeline_options.checkpoint_keep_only_best,
        pipeline_options.checkpoint_early_stop_patience,
        pipeline_options.checkpoint_validation_steps,
    )

    if isinstance(model, LinearWordQEClassifier):
        trainer = LinearWordQETrainer(
            model,
            model_options.training_algorithm,
            model_options.regularization_constant,
            checkpointer,
        )
    else:
        # Set GPU or CPU; has to be before instantiating the optimizer
        model.to(device_id)

        # Optimizer
        OptimizerClass = optimizer_class(pipeline_options.optimizer)
        optimizer = OptimizerClass(
            model.parameters(), lr=pipeline_options.learning_rate
        )
        scheduler = None
        if 0.0 < pipeline_options.learning_rate_decay < 1.0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=pipeline_options.learning_rate_decay,
                patience=pipeline_options.learning_rate_decay_start,
                verbose=True,
                mode="max",
            )

        trainer = Trainer(
            model,
            optimizer,
            checkpointer,
            log_interval=pipeline_options.log_interval,
            scheduler=scheduler,
        )
    return trainer


def retrieve_datasets(fieldset, pipeline_options, model_options, output_dir):
    """
    Creates `Dataset` objects for the training and validation sets.

    Parses files according to pipeline and model options.

    Args:
        fieldset
        pipeline_options (Namespace): Generic training options
            load_data (str): Input directory for loading preprocessed data
                files.
            load_model (str): Directory containing model.torch for loading
                pre-created model.
            resume (boolean): Indicates if you should resume training from a
                previous run.
            load_vocab (str): Directory containing vocab.torch file to be
                loaded.
        model_options (Namespace): Model specific options.
        output_dir (str): Path to directory where experiment files should be
            saved.

    Returns:
        datasets (Dataset): Training and validation datasets
    """
    if pipeline_options.load_data:
        datasets = utils.load_training_datasets(
            pipeline_options.load_data, fieldset
        )
    else:
        load_vocab = None

        if pipeline_options.resume:
            load_vocab = Path(output_dir, const.VOCAB_FILE)
        elif pipeline_options.load_model:
            load_vocab = pipeline_options.load_model
        elif model_options.__dict__.get("load_pred_source"):
            load_vocab = model_options.load_pred_source
        elif model_options.__dict__.get("load_pred_target"):
            load_vocab = model_options.load_pred_target
        elif pipeline_options.load_vocab:
            load_vocab = pipeline_options.load_vocab

        datasets = builders.build_training_datasets(
            fieldset, load_vocab=load_vocab, **vars(model_options)
        )
    return datasets


def setup(output_dir, seed=42, gpu_id=None, debug=False, quiet=False):
    """
    Analyzes pipeline options and sets up requirements for running the training
    pipeline.

    This includes setting up the output directory, random seeds and the
    device(s) where training is run.

    Args:
        output_dir: Path to directory to use or None, in which case one is
            created automatically.
        seed (int): Random seed for all random engines (Python, PyTorch, NumPy).
        gpu_id (int): GPU number to use or `None` to use the CPU.
        debug (bool): Whether to increase the verbosity of output messages.
        quiet (bool): Whether to decrease the verbosity of output messages.
            Takes precedence over `debug`.

    Returns:
        output_dir(str): Path to output directory
    """
    output_dir = setup_output_directory(
        output_dir,
        tracking_logger.run_uuid,
        tracking_logger.experiment_id,
        create=True,
    )
    configure_logging(output_dir=output_dir, debug=debug, quiet=quiet)
    configure_seed(seed)

    logging.info("This is run ID: {}".format(tracking_logger.run_uuid))
    logging.info(
        "Inside experiment ID: {} ({})".format(
            tracking_logger.experiment_id, tracking_logger.experiment_name
        )
    )
    logging.info("Local output directory is: {}".format(output_dir))
    logging.info(
        "Logging execution to MLflow at: {}".format(
            tracking_logger.get_tracking_uri()
        )
    )

    if gpu_id is not None and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        logging.info("Using GPU: {}".format(gpu_id))
    else:
        logging.info("Using CPU")

    logging.info(
        "Artifacts location: {}".format(tracking_logger.get_artifact_uri())
    )

    return output_dir


def teardown(options):
    """
    Tears down after executing prediction pipeline.

    Args:
        options(Namespace): Pipeline specific options
    """
    pass


def log(
    output_dir,
    config_options,
    config_file_name="train_config.yml",
    save_config=None,
):
    """
    Logs configuration options for the current training run.

    Args:
        output_dir (str): Path to directory where experiment files should be
            saved.
        config_options (Namespace): Namespace representing all configuration
            options.
        config_file_name (str): Filename of the config file
        save_config (str or Path): Boolean stating if you should save a
            configuration file.

    """
    logging.debug(pformat(config_options))
    config_file_copy = Path(output_dir, config_file_name)
    save_args_to_file(config_file_copy, **config_options)
    if tracking_logger.should_log_artifacts():
        tracking_logger.log_artifact(str(config_file_copy))

    if save_config:
        save_args_to_file(save_config, output_dir=output_dir, **config_options)

    # Log parameters
    tracking_logger.log_param("output_dir", output_dir)
    tracking_logger.log_param("save_config", save_config)
    for param, value in config_options.items():
        tracking_logger.log_param(param, value)
