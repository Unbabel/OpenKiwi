from pathlib import Path

import pytest

from kiwi import cli, lib


@pytest.fixture
def minimal_args(data_opts_17):
    return ['--model', 'predictor',
            '--train-source', data_opts_17.train_source,
            '--train-target', data_opts_17.train_target,
            '--valid-source', data_opts_17.valid_source,
            '--valid-target', data_opts_17.valid_target]


@pytest.fixture
def config_file_path(temp_output_dir):
    return Path(temp_output_dir) / 'config.yaml'


@pytest.fixture
def config_file_args(config_file_path):
    return ['--config', str(config_file_path)]


def test_can_write_and_read_default_opts(config_file_args, minimal_args):
    config_file_path = config_file_args[1]
    parser = cli.pipelines.train.build_parser()
    default_options = parser.parse(args=minimal_args)
    lib.utils.save_args_to_file(
        file_name=config_file_path, **vars(default_options.all_options)
    )

    try:
        loaded_options = parser.parse(config_file_args)
    except Exception as e:
        raised_exception = e
        loaded_options = None
    else:
        raised_exception = None

    assert(raised_exception is None)
    assert(default_options == loaded_options)
