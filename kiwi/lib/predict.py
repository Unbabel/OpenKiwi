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
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pydantic import validator
from typing_extensions import Literal

from kiwi.data.datasets.wmt_qe_dataset import WMTQEDataset
from kiwi.lib import evaluate
from kiwi.lib.evaluate import MetricsReport
from kiwi.lib.utils import configure_logging, configure_seed, save_config_to_file
from kiwi.runner import Runner
from kiwi.systems.outputs.quality_estimation import QEOutputs
from kiwi.systems.qe_system import BatchSizeConfig, QESystem
from kiwi.utils.io import BaseConfig, save_predicted_probabilities

logger = logging.getLogger(__name__)


class RunConfig(BaseConfig):
    seed: int = 42
    """Random seed"""

    run_id: str = None
    """If specified, MLflow/Default Logger will log metrics and params
    under this ID. If it exists, the run status will change to running.
    This ID is also used for creating this run's output directory.
    (Run ID must be a 32-character hex string)."""

    output_dir: Path = None
    """Output several files for this run under this directory.
    If not specified, a directory under "runs" is created or reused based on the
    Run UUID."""

    predict_on_data_partition: Literal['train', 'valid', 'test'] = 'test'
    """Name of the data partition to predict upon. File names are read from the
    corresponding ``data`` configuration field."""

    @validator('output_dir', always=True)
    def check_consistency(cls, v, values):
        if v is None and values.get('run_id') is None:
            raise ValueError('Must provide `run_id` or `output_dir`')
        return v


class Configuration(BaseConfig):
    run: RunConfig
    data: WMTQEDataset.Config
    system: QESystem.Config

    use_gpu: bool = False
    """If true and only if available, use the CUDA device specified in ``gpu_id`` or the
    first CUDA device. Otherwise, use the CPU."""

    gpu_id: Optional[int]
    """Use CUDA on the listed device, only if ``use_gpu`` is true."""

    verbose: bool = False
    quiet: bool = False

    @validator('system', pre=False, always=True)
    def enforce_loading(cls, v):
        if not v.load:
            raise ValueError('`system.load` is required for predicting')
        return v

    @validator('use_gpu', pre=False, always=True)
    def setup_gpu(cls, v):
        use_gpu = False
        if v:
            # Infer
            if torch.cuda.is_available():
                use_gpu = True
            else:
                logger.info('Asked to use GPU but none is available; running on CPU')

        return use_gpu

    @validator('gpu_id', pre=True, always=True)
    def setup_gpu_id(cls, v, values):
        if values.get('use_gpu'):
            if v is not None:
                return v
            else:
                logger.info(
                    f'Automatically selecting GPU {torch.cuda.current_device()}'
                )
                return torch.cuda.current_device()
        return None


def load_system(system_path: Union[str, Path], gpu_id: Optional[int] = None):
    """Load a pretrained system (model) into a `Runner` object.

    Args:
      system_path: A path to the saved checkpoint file produced by a training run.
      gpu_id: id of the gpu to load the model into (-1 or None to use CPU)

    Throws:
      Exception: If the path does not exist, or is not a valid system file.

    """
    system_path = Path(system_path)
    if not system_path.exists():
        raise Exception(f'Path "{system_path}" does not exist!')

    system = QESystem.load(system_path)
    if not system:
        raise Exception(f'No model found in "{system_path}"')

    return Runner(system, device=gpu_id)


def predict_from_configuration(configuration_dict: Dict[str, Any]):
    """Run the entire prediction pipeline using the configuration options received."""
    config = Configuration(**configuration_dict)

    logger.debug('Setting up predict..')
    output_dir = setup_run(
        config.run,
        quiet=configuration_dict['quiet'],
        debug=configuration_dict['verbose'],
    )
    logging.debug(pformat(config.dict()))
    config_file = output_dir / 'predict_config.yaml'
    save_config_to_file(config, config_file)

    logger.debug('Predict set up. Running...')
    predictions, metrics = run(config, output_dir)
    if metrics:
        logger.info(f'Evaluation metrics (since there were gold files):\n{metrics}')


def run(
    config: Configuration, output_dir: Path
) -> Tuple[Dict[str, List], Optional[MetricsReport]]:
    """Run the prediction pipeline.

    Load the model and necessary files and create the model's predictions for the
    configured data partition.

    Arguments:
        config: validated configuration values for the (predict) pipeline.
        output_dir: directory where to save predictions.

    Return:
        Predictions: Dictionary with format {'target': predictions}
    """
    outputs_config = None
    if config.system.model and 'outputs' in config.system.model:
        outputs_config = QEOutputs.Config(**config.system.model['outputs'])

    predictions = make_predictions(
        output_dir=output_dir,
        best_model_path=config.system.load,
        data_partition=config.run.predict_on_data_partition,
        data_config=config.data,
        outputs_config=outputs_config,
        batch_size=config.system.batch_size,
        num_workers=config.system.num_data_workers,
        gpu_id=config.gpu_id,
    )

    metrics = None
    if config.run.predict_on_data_partition == 'valid' and config.data.valid.output:
        eval_config = evaluate.Configuration(
            gold_files=config.data.valid.output, predicted_dir=output_dir,
        )
        metrics = evaluate.run(eval_config)

    return predictions, metrics


def make_predictions(
    output_dir: Path,
    best_model_path: Path,
    data_partition: Literal['train', 'valid', 'test'],
    data_config: WMTQEDataset.Config,
    outputs_config: QEOutputs.Config = None,
    batch_size: Union[int, BatchSizeConfig] = None,
    num_workers: int = 0,
    gpu_id: int = None,
):
    """Make predictions over the validation set using the best model created during
    training.

    Arguments:
        output_dir: output Directory where predictions should be saved.
        best_model_path: path pointing to the checkpoint with best performance.
        data_partition: on which dataset to predict (one of 'train', 'valid', 'test').
        data_config: configuration containing options for the ``data_partition`` set.
        outputs_config: configuration specifying which outputs to activate.
        batch_size: for predicting.
        num_workers: number of parallel data loaders.
        gpu_id: GPU to use for predicting; 0 for CPU.

    Return:
        dictionary with predictions in the format {'target': predictions}.
    """
    runner = load_system(best_model_path, gpu_id=gpu_id)
    runner.system.set_config_options(
        data_config=data_config, batch_size=batch_size, num_data_workers=num_workers,
    )
    if outputs_config:
        runner.configure_outputs(outputs_config)

    dataloader = None
    if data_partition == 'train':
        dataloader = runner.system.train_dataloader()
    elif data_partition == 'valid':
        dataloader = runner.system.val_dataloader()
    elif data_partition == 'test':
        dataloader = runner.system.test_dataloader()

    logger.info(
        f'Predicting with the {type(runner.system).__name__} model over the '
        f'{data_partition or "test"} set.'
    )

    predictions = runner.run(dataloader)
    save_predicted_probabilities(output_dir, predictions)

    return predictions


def setup_run(
    config: RunConfig, quiet=False, debug=False, anchor_dir: Path = None
) -> Path:
    """Prepare for running the prediction pipeline.

    This includes setting up the output directory, random seeds, and loggers.

    Arguments:
        config: configuration options.
        quiet: whether to suppress info log messages.
        debug: whether to additionally log debug messages
               (:param:`quiet` has precedence)
        anchor_dir: directory to use as root for paths.

    Return:
        the resolved path to the output directory.
    """
    # Setup output directory
    output_dir = config.output_dir
    if not output_dir:
        experiment_id = 0
        run_id = config.run_id or uuid.uuid4().hex  # Create hash if needed
        output_dir = Path('runs') / str(experiment_id) / run_id
        if anchor_dir:
            output_dir = anchor_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir=output_dir, verbose=debug, quiet=quiet)
    configure_seed(config.seed)

    logger.info(f'Local output directory is: {output_dir}')

    return output_dir
