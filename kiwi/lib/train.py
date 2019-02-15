import logging
from pathlib import Path
from pprint import pformat

import torch

from kiwi import constants as const
from kiwi.cli.pipelines.train import build_parser
from kiwi.data import builders, utils
from kiwi.data.iterators import build_bucket_iterator
from kiwi.data.utils import save_training_datasets, save_vocabularies_from_datasets
from kiwi.lib.utils import (
    configure_logging,
    configure_seed,
    merge_namespaces,
    save_args_to_file,
    setup_output_directory,
)
from kiwi.loggers import mlflow_logger
from kiwi.models.linear_word_qe_classifier import LinearWordQEClassifier
from kiwi.models.model import Model
from kiwi.trainers.callbacks import Checkpoint
from kiwi.trainers.linear_word_qe_trainer import LinearWordQETrainer
from kiwi.trainers.trainer import Trainer
from kiwi.trainers.utils import optimizer_class

logger = logging.getLogger(__name__)


class TrainRunInfo:
    """
    Class used to encapsulate relevant information on training runs.

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
        self.run_uuid = mlflow_logger.run_uuid


def train_from_file(filename):
    """
    Imports and parses a config file and calls `train_from_options`.

    Args:
        filename (str): filename of the configuration file
    """
    parser = build_parser()
    options = parser.parse_config_file(filename)
    return train_from_options(options)


def train_from_options(options):
    """
    Runs the entire training pipeline using the configuration options
    received. These options include the pipeline and model options
    plus the model's API.

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

    mlflow_run = mlflow_logger.configure(
        run_uuid=pipeline_options.run_uuid,
        experiment_name=pipeline_options.experiment_name,
        tracking_uri=pipeline_options.mlflow_tracking_uri,
        always_log_artifacts=pipeline_options.mlflow_always_log_artifacts,
    )

    with mlflow_run:
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
            save_config=pipeline_options.save_config,
            config_options=vars(all_options),
        )

        trainer = run(ModelClass, output_dir, pipeline_options, model_options)
        train_info = TrainRunInfo(trainer)

    teardown(pipeline_options)

    return train_info


def run(ModelClass, output_dir, pipeline_options, model_options):
    """

    Args:
        ModelClass (type): Python Type of the Model to train
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

    datasets = retrieve_datasets(fieldset, pipeline_options, model_options, output_dir)
    save_vocabularies_from_datasets(output_dir, *datasets)
    if pipeline_options.save_data:
        save_training_datasets(pipeline_options.save_data, *datasets)

    # Trainer step
    device_id = None
    if pipeline_options.gpu_id is not None and pipeline_options.gpu_id >= 0:
        device_id = pipeline_options.gpu_id

    vocabs = utils.fields_to_vocabs(datasets[0].fields)

    trainer = retrieve_trainer(
        ModelClass, pipeline_options, model_options, vocabs, output_dir, device_id
    )

    logger.info(str(trainer.model))
    logger.info("{} parameters".format(trainer.model.num_parameters()))

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
    """Create Trainer object.
       Method creates Trainer, Model and Checkpointer
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
    """
    if pipeline_options.load_data:
        datasets = utils.load_training_datasets(pipeline_options.load_data, fieldset)
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
    Analyze pipeline options and set up requirements to running
    the training pipeline. This includes setting up the output
    directory, random seeds and the device(s) where training is run.

    Args:
        options(Namespace): Pipeline specific options

    Returns:
        output_dir(str): Path to output directory
    """
    output_dir = setup_output_directory(
        output_dir, mlflow_logger.run_uuid, mlflow_logger.experiment_id, create=True
    )
    configure_logging(output_dir=output_dir, debug=debug, quiet=quiet)
    configure_seed(seed)

    logging.info("This is run ID: {}".format(mlflow_logger.run_uuid))
    logging.info(
        "Inside experiment ID: {} ({})".format(
            mlflow_logger.experiment_id, mlflow_logger.experiment_name
        )
    )
    logging.info("Local output directory is: {}".format(output_dir))
    logging.info(
        "Logging execution to MLflow at: {}".format(mlflow_logger.get_tracking_uri())
    )

    if gpu_id is not None and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        logging.info("Using GPU: {}".format(gpu_id))
    else:
        logging.info("Using CPU")

    logging.info("Artifacts location: {}".format(mlflow_logger.get_artifact_uri()))

    return output_dir


def teardown(options):
    """
    Tears down after executing prediction pipeline.

    Args:
        options(Namespace): Pipeline specific options
    """
    pass
    pass


def log(output_dir, save_config, config_options, config_file_name="train_config.yml"):
    logging.debug(pformat(config_options))
    config_file_copy = Path(output_dir, config_file_name)
    save_args_to_file(config_file_copy, **config_options)
    if mlflow_logger.should_log_artifacts():
        mlflow_logger.log_artifact(str(config_file_copy))

    if save_config:
        save_args_to_file(save_config, output_dir=output_dir, **config_options)

    # Log parameters
    mlflow_logger.log_param("output_dir", output_dir)
    mlflow_logger.log_param("save_config", save_config)
    for param, value in config_options.items():
        mlflow_logger.log_param(param, value)
