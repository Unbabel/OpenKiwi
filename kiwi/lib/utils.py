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
import json
import logging
from pathlib import Path
from time import gmtime
from typing import Dict, Union

import yaml
from pytorch_lightning import seed_everything

from kiwi.utils.io import BaseConfig


def configure_seed(seed: int):
    """Configure the random seed for all relevant packages.

    These include: random, numpy, torch, torch.cuda and PYTHONHASHSEED.

    Arguments:
        seed: the random seed to be set.
    """
    seed_everything(seed)


def configure_logging(
    output_dir: Path = None, verbose: bool = False, quiet: bool = False
):
    """Configure the output logger.

    Set up the log format, logging level, and output directory of logging.

    Arguments:
        output_dir: the directory where log output will be stored; defaults to None.
        verbose: change logging level to debug.
        quiet: change logging level to warning to suppress info logs.
    """
    logging.Formatter.converter = gmtime
    date_format = '%Y-%m-%dT%H:%M:%SZ'
    log_format = '%(asctime)s %(levelname)-8s %(name)24.24s:%(lineno)3.3s: %(message)s'
    if logging.getLogger().handlers:
        log_formatter = logging.Formatter(log_format, datefmt=date_format)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(log_formatter)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)

    log_level = logging.INFO
    if verbose:
        log_level = logging.DEBUG
    if quiet:
        log_level = logging.WARNING
    logging.getLogger().setLevel(log_level)

    if output_dir is not None:
        fh = logging.FileHandler(str(output_dir / 'output.log'))
        fh.setLevel(log_level)
        logging.getLogger().addHandler(fh)

    # Silence urllib3
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)


def save_config_to_file(config: BaseConfig, file_name: Union[str, Path]):
    """Save a configuration object to file.

    Arguments:
        file_name: where to saved the configuration.
        config: a pydantic configuration object.
    """
    path = Path(file_name).with_suffix('.yaml')
    yaml.dump(json.loads(config.json()), path.open('w'))
    logging.debug(f'Saved current options to config file: {path}')


def setup_output_directory(
    output_dir, run_uuid=None, experiment_id=None, create=True
) -> str:
    """Set up the output directory.

    This means either creating one, or verifying that the provided directory exists.
    Output directories are created using the run and experiment ids.

    Arguments:
        output_dir: the target output directory.
        run_uuid : the hash of the current run.
        experiment_id: the id of the current experiment.
        create: whether to create the directory.

    Return:
        the path to the resolved output directory.
    """
    if not output_dir:
        if experiment_id is None or run_uuid is None:
            raise ValueError('No output directory or run_uuid have been specified.')
        output_path = Path('runs', str(experiment_id), str(run_uuid))
        output_dir = str(output_path)

    if create:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    elif not Path(output_dir).exists():
        raise FileNotFoundError(
            'Output directory does not exist: {}'.format(output_dir)
        )

    return output_dir


def load_config(config_file: Path) -> Dict:
    """Load configuration options from a YAML or JSON file."""
    if not config_file.exists():
        raise FileNotFoundError(f"'{config_file}' does not exist")
    with config_file.open() as f:
        if config_file.suffix == '.json':
            import json

            config_dict = json.load(f)
        elif config_file.suffix == '.yaml' or config_file.suffix == 'yml':
            import yaml

            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise TypeError(f'Unsupported config file format: {config_file.suffix}')
    return config_dict
