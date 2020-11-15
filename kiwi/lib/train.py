#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2020 Unbabel <openkiwi@unbabel.com>
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
import uuid
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from pydantic import PositiveInt, validator
from pydantic.types import confloat, conint
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing_extensions import Literal

from kiwi.data.datasets.wmt_qe_dataset import WMTQEDataset
from kiwi.lib import evaluate
from kiwi.lib.predict import load_system
from kiwi.lib.utils import (
    configure_logging,
    configure_seed,
    file_to_configuration,
    save_config_to_file,
)
from kiwi.loggers import MLFlowTrackingLogger
from kiwi.systems.qe_system import QESystem
from kiwi.systems.tlm_system import TLMSystem
from kiwi.training.callbacks import BestMetricsInfo
from kiwi.utils.io import BaseConfig, save_predicted_probabilities

logger = logging.getLogger(__name__)


@dataclass
class TrainRunInfo:
    """Encapsulate relevant information on training runs."""

    model: QESystem
    """The last model when training finished."""

    best_metrics: Dict[str, Any]
    """Mapping of metrics of the best model."""

    best_model_path: Optional[Path] = None
    """Path of the best model, if it was saved to disk."""


class RunConfig(BaseConfig):
    """Options for each run."""

    seed: int = 42
    """Random seed"""

    experiment_name: str = 'default'
    """If using MLflow, it will log this run under this experiment name, which appears
    as a separate section in the UI. It will also be used in some messages and files."""

    output_dir: Path = None
    """Output several files for this run under this directory.
    If not specified, a directory under "./runs/" is created or reused based on the
    ``run_id``. Files might also be sent to MLflow depending on the
    ``mlflow_always_log_artifacts`` option."""

    run_id: str = None
    """If specified, MLflow/Default Logger will log metrics and params
    under this ID. If it exists, the run status will change to running.
    This ID is also used for creating this run's output directory if
    ``output_dir`` is not specified (Run ID must be a 32-character hex string)."""

    use_mlflow: bool = False
    """Whether to use MLflow for tracking this run. If not installed, a message
    is shown"""

    mlflow_tracking_uri: str = 'mlruns/'
    """If using MLflow, logs model parameters, training metrics, and
    artifacts (files) to this MLflow server. Uses the localhost by
    default. """

    mlflow_always_log_artifacts: bool = False
    """If using MLFlow, always log (send) artifacts (files) to MLflow
    artifacts URI. By default (false), artifacts are only logged if
    MLflow is a remote server (as specified by --mlflow-tracking-uri
    option).All generated files are always saved in --output-dir, so it
    might be considered redundant to copy them to a local MLflow
    server. If this is not the case, set this option to true."""


class CheckpointsConfig(BaseConfig):
    validation_steps: Union[confloat(gt=0.0, le=1.0), PositiveInt] = 1.0
    """How often within one training epoch to check the validation set.
    If float, % of training epoch. If int, check every n batches."""

    save_top_k: int = 1
    """Save and keep only ``k`` best models according to main metric;
    -1 will keep all; 0 will never save a model."""

    early_stop_patience: conint(ge=0) = 0
    """Stop training if evaluation metrics do not improve after X validations;
    0 disables this."""


class GPUConfig(BaseConfig):
    gpus: Union[int, List[int]] = 0
    """Use the number of GPUs specified if int, where 0 is no GPU. -1 is all GPUs.
    Alternatively, if a list, uses the GPU-ids specified (e.g., [0, 2])."""

    precision: Literal[16, 32] = 32
    """The floating point precision to be used while training the model. Available
    options are 32 or 16 bits."""

    amp_level: Literal['O0', 'O1', 'O2', 'O3'] = 'O0'
    """The automatic-mixed-precision level to use. O0 is FP32 training. 01 is mixed
    precision training as popularized by NVIDIA Apex. O2 casts the model weights to FP16
     but keeps certain master weights and batch norm in FP32 without patching Torch
    functions. 03 is full FP16 training."""

    @validator('gpus', pre=False, always=True)
    def setup_gpu_ids(cls, v):
        """If asking to use CPU, let it be, outputting a warning if GPUs are available.
        If asking to use any GPU but none are available, fall back to CPU and warn user.
        """
        import torch

        if v == 0:
            if torch.cuda.is_available():
                logger.info(
                    f'Using CPU for training but there are {torch.cuda.device_count()} '
                    f'GPUs available; set `trainer.gpus=-1` to use them.'
                )
        else:
            if not torch.cuda.is_available():
                logger.warning(
                    f'Asked to use GPUs for training but none are available; '
                    f'falling back to CPU (configuration was `trainer.gpus={v}`).'
                )
                v = 0

        return v

    @validator('amp_level', always=True)
    def setup_amp_level(cls, v, values):
        """If precision is set to 16, amp_level needs to be greater than O0.
        Following the same logic, if amp_level is set to greater than O0, precision
        needs to be set to 16."""

        if values.get('precision') == 16 and v == ['O0']:
            logger.warning(
                'Precision set to FP16 but AMP_level set to O0. Setting to '
                'O1 mixed precision training.'
            )
            return 'O1'
        elif v in ['O1', 'O2', 'O3'] and values.get('precision') == 32:
            logger.warning(
                f'Precision set to FP32 but AMP_level set to {v}. Setting to '
                'O0 full precision training.'
            )
            return 'O0'
        return v


class TrainerConfig(GPUConfig):
    resume: bool = False
    """Resume training a previous run.
    The `run.run_id` (and possibly `run.experiment_name`) option must be specified.
    Files are then searched under the "runs" directory. If not found, they are
    downloaded from the MLflow server (check the `mlflow_tracking_uri` option)."""

    epochs: int = 50
    """Number of epochs for training."""

    gradient_accumulation_steps: int = 1
    """Accumulate gradients for the given number of steps (batches) before
        back-propagating."""

    gradient_max_norm: float = 0.0
    """Clip gradients with norm above this value; by default (0.0), do not clip."""

    main_metric: Union[str, List[str]] = None
    """Choose Primary Metric for this run."""

    log_interval: int = 100
    """Log every k batches."""

    log_save_interval: int = 100
    """Save accumulated log every k batches (does not seem to
    matter to MLflow logging)."""

    checkpoint: CheckpointsConfig = CheckpointsConfig()

    deterministic: bool = True
    """If true enables cudnn.deterministic. Might make training slower, but ensures
     reproducibility."""


class Configuration(BaseConfig):
    run: RunConfig
    """Options specific to each run"""

    trainer: TrainerConfig
    data: WMTQEDataset.Config
    system: QESystem.Config

    debug: bool = False
    """Run training in `fast_dev` mode; only one batch is used for training and
    validation. This is useful to test out new models."""

    verbose: bool = False
    quiet: bool = False


def train_from_file(filename) -> TrainRunInfo:
    """Load options from a config file and calls the training procedure.

    Arguments:
        filename: of the configuration file.

    Return:
        an object with training information.
    """
    config = file_to_configuration(filename)
    return train_from_configuration(config)


def train_from_configuration(configuration_dict) -> TrainRunInfo:
    """Run the entire training pipeline using the configuration options received.

    Arguments:
        configuration_dict: dictionary with options.

    Return: object with training information.
    """
    config = Configuration(**configuration_dict)

    train_info = run(config)

    return train_info


def setup_run(
    config: RunConfig, debug=False, quiet=False, anchor_dir: Path = None
) -> Tuple[Path, Optional[MLFlowTrackingLogger]]:
    """Prepare for running the training pipeline.

    This includes setting up the output directory, random seeds, and loggers.

    Arguments:
        config: configuration options.
        quiet: whether to suppress info log messages.
        debug: whether to additionally log debug messages
               (:param:`quiet` has precedence)
        anchor_dir: directory to use as root for paths.

    Return:
         a tuple with the resolved path to the output directory and the experiment
         logger (``None`` if not configured).
    """

    # Setup tracking logger
    if config.use_mlflow:
        tracking_logger = MLFlowTrackingLogger(
            experiment_name=config.experiment_name,
            run_id=config.run_id,
            tracking_uri=config.mlflow_tracking_uri,
            always_log_artifacts=config.mlflow_always_log_artifacts,
        )
        experiment_id = tracking_logger.experiment_id
        run_id = tracking_logger.run_id
    else:
        tracking_logger = None
        experiment_id = 0
        run_id = config.run_id or uuid.uuid4().hex  # Create hash if needed

    # Setup output directory
    output_dir = config.output_dir
    if not output_dir:
        output_dir = Path('runs') / str(experiment_id) / run_id
        if anchor_dir:
            output_dir = anchor_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir=output_dir, verbose=debug, quiet=quiet)
    configure_seed(config.seed)

    logging.info(f'This is run ID: {run_id}')
    logging.info(
        f'Inside experiment ID: ' f'{experiment_id} ({config.experiment_name})'
    )
    logging.info(f'Local output directory is: {output_dir}')

    if tracking_logger:
        logging.info(f'Logging execution to MLFlow at: {tracking_logger.tracking_uri}')
        logging.info(f'Artifacts location: {tracking_logger.get_artifact_uri()}')

    return output_dir, tracking_logger


def run(
    config: Configuration,
    system_type: Union[Type[TLMSystem], Type[QESystem]] = QESystem,
) -> TrainRunInfo:
    """Instantiate the system according to the configuration and train it.

    Load or create a trainer for doing it.

    Args:
        config: generic training options.
        system_type: class of system being used.

    Return:
        an object with training information.
    """
    output_dir, tracking_logger = setup_run(
        config.run, debug=config.verbose, quiet=config.quiet,
    )

    # Log configuration options for the current training run
    logging.debug(pformat(config.dict()))
    config_file = output_dir / 'train_config.yaml'
    save_config_to_file(config, config_file)
    if tracking_logger:
        tracking_logger.log_artifact(config_file)
        tracking_logger.log_param('output_dir', output_dir)
        tracking_logger.log_hyperparams(config.dict())

    # Instantiate system (i.e., model)
    system = system_type.from_config(config.system, data_config=config.data)

    logging.info(f'Training the {config.system.class_name} model')
    logging.info(str(system))
    logging.info(f'{system.num_parameters()} parameters')
    if tracking_logger:
        model_description = (
            f"## Number of parameters: {system.num_parameters()}\n\n"
            f"## Model architecture\n"
            f"```\n{system}\n```\n"
        )
        tracking_logger.log_tag('mlflow.note.content', model_description)

    metric_name, metric_ordering = system.main_metric(config.trainer.main_metric)
    metric_name = f'val_{metric_name}'
    checkpoint_callback = ModelCheckpoint(
        filepath=str(
            output_dir / f'checkpoints/model_{{epoch:02d}}-{{{metric_name}:.2f}}'
        ),
        monitor=metric_name,
        mode=metric_ordering,
        save_top_k=config.trainer.checkpoint.save_top_k,
        save_weights_only=True,
        verbose=True,
        period=0,  # Always allow saving checkpoint even within the same epoch
    )
    early_stop_callback = EarlyStopping(
        monitor=metric_name,
        mode=metric_ordering,
        patience=config.trainer.checkpoint.early_stop_patience,
        verbose=True,
    )
    best_metrics_callback = BestMetricsInfo(monitor=metric_name, mode=metric_ordering)

    trainer = pl.Trainer(
        logger=tracking_logger or False,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        callbacks=[best_metrics_callback],
        gpus=config.trainer.gpus,
        #
        max_epochs=config.trainer.epochs,
        min_epochs=1,
        #
        check_val_every_n_epoch=1,
        val_check_interval=config.trainer.checkpoint.validation_steps,
        #
        accumulate_grad_batches=config.trainer.gradient_accumulation_steps,
        gradient_clip_val=config.trainer.gradient_max_norm,
        #
        progress_bar_refresh_rate=logger.isEnabledFor(logging.INFO),
        log_save_interval=config.trainer.log_save_interval,
        row_log_interval=config.trainer.log_interval,
        # Debugging and informative flags
        log_gpu_memory='min_max',
        weights_summary=(None if config.quiet else 'full'),
        #
        num_sanity_val_steps=5,
        deterministic=config.trainer.deterministic,
        fast_dev_run=config.debug,
        # For eventual extra performance
        amp_level=config.trainer.amp_level,
        precision=config.trainer.precision,
        #
    )
    trainer.fit(system)

    # Get best model path in case there have been checkpoints saved
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        logger.warning(
            'No checkpoint was saved. Exiting gracefully and returning training info.'
        )
        run_info = TrainRunInfo(
            model=trainer.model,
            best_metrics=best_metrics_callback.best_metrics,
            best_model_path=None,
        )
        return run_info

    if trainer.model.config.model.encoder.adapter is not None:
        if trainer.model.config.model.encoder.adapter.fusion:
            # We just save the entire model
            pass
        else:
            language = trainer.model.config.model.encoder.adapter.language
            adapter_path = (
                Path(checkpoint_callback.best_model_path).parent / f'{language}'
            )
            logger.info(f"Saving the Adapter '{language}' to: {adapter_path}")
            if 'bert' in trainer.model.encoder.__dict__.keys():
                trainer.model.encoder.bert.save_adapter(adapter_path, language)
            elif 'xlm_roberta' in trainer.model.encoder.__dict__.keys():
                trainer.model.encoder.xlm_roberta.save_adapter(adapter_path, language)

    if tracking_logger:
        # Send best model file to logger
        tracking_logger.log_model(best_model_path)

    # Load best model and predict
    if system_type == QESystem:
        # TLMSystems don't need to create predictions over the validation set
        logger.info(
            'Finished training. Using best checkpoint to make predictions on the '
            'validation set (and test set, if configured).'
        )
        runner = load_system(
            best_model_path,
            gpu_id=None if config.trainer.gpus == 0 else torch.cuda.current_device(),
        )
        data_config = config.data.copy()
        data_config.train = None  # Avoid loading the train dataset
        runner.system.set_config_options(data_config=data_config)

        # Get and save predictions on the validation set
        predictions = runner.run(runner.system.val_dataloader())
        save_predicted_probabilities(output_dir, predictions)
        # Run evaluation and report it
        eval_config = evaluate.Configuration(
            gold_files=config.data.valid.output, predicted_dir=output_dir,
        )
        metrics = evaluate.run(eval_config)
        logger.info(f'Evaluation on the validation set:\n{metrics}')

        if config.data.test:
            logger.info('Predicting on the test set...')
            predictions = runner.run(runner.system.test_dataloader())
            save_predicted_probabilities(output_dir / 'test', predictions)

    run_info = TrainRunInfo(
        model=trainer.model,
        best_metrics=best_metrics_callback.best_metrics,
        best_model_path=best_model_path,
    )

    return run_info
