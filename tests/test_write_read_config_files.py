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

from pathlib import Path

import pytest

from kiwi.cli.pipelines.train import build_parser
from kiwi.lib.utils import (
    parse_integer_with_positive_infinity,
    save_args_to_file,
)


@pytest.fixture
def minimal_args(data_opts_17):
    return [
        '--model',
        'predictor',
        '--train-source',
        data_opts_17.train_source,
        '--train-target',
        data_opts_17.train_target,
        '--valid-source',
        data_opts_17.valid_source,
        '--valid-target',
        data_opts_17.valid_target,
    ]


@pytest.fixture
def config_file_path(temp_output_dir):
    return Path(temp_output_dir) / 'config.yaml'


@pytest.fixture
def config_file_args(config_file_path):
    return ['--config', str(config_file_path)]


def test_can_write_and_read_default_opts(config_file_args, minimal_args):
    config_file_path = config_file_args[1]
    parser = build_parser()
    default_options = vars(parser.parse(args=minimal_args).all_options)
    save_args_to_file(file_name=config_file_path, **default_options)

    try:
        loaded_options = parser.parse(config_file_args).all_options
    except Exception as e:
        raised_exception = e
        loaded_options = None
    else:
        raised_exception = None

    assert raised_exception is None
    assert loaded_options.source_max_length == float('inf')
    assert loaded_options.target_max_length == float('inf')

    loaded_options = vars(loaded_options)
    del loaded_options['config']
    del default_options['config']
    for key, value in loaded_options.items():
        assert key in default_options
        assert value == default_options[key]
        del default_options[key]
    assert not default_options


def test_parse_int_with_infinity_returns_int():
    for integer in range(-10, 10):
        assert parse_integer_with_positive_infinity(str(integer)) == integer


def test_parse_int_with_infinity_returns_infinity():
    infinity = float('inf')
    assert parse_integer_with_positive_infinity(str(infinity)) == infinity


def test_parse_int_with_infinity_raises_value_error():
    not_infinity = 'not_infinity'
    with pytest.raises(ValueError):
        parse_integer_with_positive_infinity(str(not_infinity))
