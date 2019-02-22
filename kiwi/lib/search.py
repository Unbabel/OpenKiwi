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

import itertools
from collections import OrderedDict
from pathlib import Path

import configargparse

from kiwi.cli.opts import PathType
from kiwi.lib import train
from kiwi.models.model import Model

parser = configargparse.get_argument_parser('search')

parser.add_argument(
    '-e',
    '--experiment-name',
    required=False,
    help='MLflow will log this run under this experiment name, '
    'which appears as a separate section in the UI. It '
    'will also be used in some messages and files.',
)
parser.add(
    '-c',
    '--config',
    required=True,
    is_config_file=False,
    type=PathType(exists=True),
    help='Load config file from path',
)

group = parser.add_argument_group('models')
group.add_argument('model_name', choices=Model.subclasses.keys())


def get_action(option):
    for action in train.parser._actions:
        if option in train.parser.get_possible_config_keys(action):
            return action
    return None


def split_options(options):
    meta_options = OrderedDict()
    normal_options = []
    for key, value in options.items():
        if isinstance(value, list):
            meta_options[key] = value
        else:
            action = get_action(key)
            normal_options += parser.convert_item_to_command_line_arg(
                action, key, value
            )
    return meta_options, normal_options


def run(options, extra_options):
    config_parser = configargparse.YAMLConfigFileParser()
    config_options = config_parser.parse(Path(options.config).read_text())
    meta, fixed_options = split_options(config_options)

    # Run for each combination of arguments
    fixed_args = [options.model_name] + extra_options
    if options.experiment_name:
        fixed_args += parser.convert_item_to_command_line_arg(
            None, 'experiment-name', options.experiment_name
        )

    meta_keys = meta.keys()
    meta_values = meta.values()
    for values in itertools.product(*meta_values):
        assert len(meta_keys) == len(values)
        run_args = []
        for key, value in zip(meta_keys, values):
            action = get_action(key)
            run_args.extend(
                parser.convert_item_to_command_line_arg(action, key, str(value))
            )
        full_args = fixed_args + run_args + fixed_options
        train.main(full_args)


def main(argv=None, external_options=None):
    raise NotImplementedError('Pipeline not yet supported.')

    # options, extra_options = parser.parse_known_args(args=argv)
    # run(options, extra_options)


if __name__ == '__main__':
    main()
