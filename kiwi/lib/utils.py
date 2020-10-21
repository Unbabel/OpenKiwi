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
import os.path
from pathlib import Path
from time import gmtime
from typing import Dict, Union

import hydra.experimental
import hydra.utils
import yaml
from hydra._internal.hydra import Hydra
from omegaconf import OmegaConf
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


def file_to_configuration(config_file: Union[str, Path]) -> Dict:
    return arguments_to_configuration({'CONFIG_FILE': config_file})


def arguments_to_configuration(arguments: Dict) -> Dict:
    config_file = Path(arguments['CONFIG_FILE'])

    # Using Hydra
    relative_dir = Path(
        os.path.relpath(config_file.resolve().parent, start=Path(__file__).parent)
    )
    Hydra.create_main_hydra_file_or_module(
        calling_file=__file__,
        calling_module=None,
        config_dir=str(relative_dir),
        strict=False,
    )
    config = hydra.experimental.compose(
        config_file=config_file.name, overrides=arguments.get('OVERWRITES', [])
    )
    # print(config.pretty())

    # Back to a dictionary
    config_dict = OmegaConf.to_container(config)

    # config_dict['anchor_directory'] = config_file.parent
    config_dict['verbose'] = arguments.get('--verbose', False)
    config_dict['quiet'] = arguments.get('--quiet', False)

    return config_dict
