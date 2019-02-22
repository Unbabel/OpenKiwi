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

import argparse
import logging
import random
from argparse import Namespace
from pathlib import Path
from time import gmtime

import configargparse
import numpy as np
import torch


def configure_seed(seed):
    """
    Configure the random seed for all relevant packages.
    These include: random, numpy, torch and torch.cuda

    Args:
        seed (int): the random seed to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def configure_device(gpu_id):
    """
    Configure gpu to be used in computation.

    Args:
        gpu_id (int): The id of the gpu to be used
    """
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)


def configure_logging(output_dir=None, debug=False, quiet=False):
    """
    Configure the logger. Sets up the log format, logging level
    and output directory of logging.

    Args:
        output_dir: The directory where log output will be stored.
            Defaults to None.
        debug (bool): Change logging level to debug.
        quiet (bool): Change logging level to warning to supress info logs.
    """
    logging.Formatter.converter = gmtime
    logging.Formatter.default_msec_format = '%s.%03d'
    log_format = '%(asctime)s [%(name)s %(funcName)s:%(lineno)s] %(message)s'
    if logging.getLogger().handlers:
        log_formatter = logging.Formatter(log_format)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(log_formatter)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
    if quiet:
        log_level = logging.WARNING

    logging.getLogger().setLevel(log_level)
    if output_dir is not None:
        fh = logging.FileHandler(str(Path(output_dir, 'output.log')))
        fh.setLevel(log_level)
        logging.getLogger().addHandler(fh)


def save_args_to_file(file_name, **kwargs):
    """
    Saves `**kwargs` to a file.

    Args:
        file_name (str): The name of the file where the args should
            be saved in.

    """
    options_to_save = {
        k.replace('_', '-'): v for k, v in kwargs.items() if v is not None
    }

    content = configargparse.YAMLConfigFileParser().serialize(options_to_save)
    Path(file_name).write_text(content)
    logging.debug('Saved current options to config file: {}'.format(file_name))


def save_config_file(options, file_name):
    """
    Saves a configuration file with OpenKiwi configuration options.
    Calls `save_args_to_file`.

    Args:
        options (Namespace): Namespace with all configuration options
            that should be saved.
        file_name (str): Name of the output configuration file.
    """
    # parser.write_config_file(options, [file_name], exit_after=False)
    save_args_to_file(file_name, **vars(options))


def setup_output_directory(
    output_dir, run_uuid=None, experiment_id=None, create=True
):
    """
    Sets up the output directory. This means either creating one, or
    verifying that the provided directory exists. Output directories
    are created using the run and experiment ids.

    Args:
        output_dir (str): The target output directory
        run_uuid : The current hash of the current run.
        experiment_id: The id of the current experiment
        create (bool): Boolean indicating whether to create a new folder.
    """
    if not output_dir:
        if experiment_id is None or run_uuid is None:
            raise argparse.ArgumentError(
                message='Please specify an output directory (--output-dir).',
                argument=output_dir,
            )
        output_path = Path('runs', str(experiment_id), str(run_uuid))
        output_dir = str(output_path)

    if create:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    elif not Path(output_dir).exists():
        raise FileNotFoundError(
            'Output directory does not exist: {}'.format(output_dir)
        )

    return output_dir


def merge_namespaces(*args):
    """
    Utility function used to merge Namespaces. Useful for merging Argparse
    options.

    Args:
        *args: Variable length list of Namespaces
    """
    if not args:
        return None
    options = {}
    for arg in filter(None, args):
        options.update(dict(vars(arg)))
    return Namespace(**options)


def parse_integer_with_positive_infinity(string):
    """
    Workaround to be able to pass both integers and infinity as CLAs.

    Args:
        string: A string representation of an integer, or infinity

    """
    try:
        integer = int(string)
        return integer
    except ValueError:
        infinity = float(string)
        if infinity == float('inf'):
            return infinity

    raise ValueError(
        'Could not parse argument "{}" as integer'
        ' with positive infinity'.format(string)
    )
