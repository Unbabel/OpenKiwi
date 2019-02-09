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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def configure_device(gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)


def configure_logging(output_dir=None, debug=False, quiet=False):
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
    options_to_save = {
        k.replace('_', '-'): v for k, v in kwargs.items() if v is not None
    }

    content = configargparse.YAMLConfigFileParser().serialize(options_to_save)
    Path(file_name).write_text(content)
    logging.debug('Saved current options to config file: {}'.format(file_name))


def save_config_file(options, file_name):
    # parser.write_config_file(options, [file_name], exit_after=False)
    save_args_to_file(file_name, **vars(options))


def setup_output_directory(
    output_dir, run_uuid=None, experiment_id=None, create=True
):
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
    if not args:
        return None
    options = {}
    for arg in filter(None, args):
        options.update(dict(vars(arg)))
    return Namespace(**options)
