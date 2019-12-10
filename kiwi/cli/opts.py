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
from distutils.util import strtobool
from pathlib import Path

from kiwi import constants as const


class PathType(object):
    """Factory for creating pathlib.Path objects

    Instances of PathType should passed as type= arguments to the
    ArgumentParser add_argument() method.

    Strongly based on argparse.FileType.

    Keyword Arguments:
        - exists -- Whether the file must exists or not.

    """

    def __init__(self, exists=False):
        self._must_exist = exists

    def __call__(self, string):
        if not string:
            return string

        # The special argument "-" means sys.std{in,out} in argparse.FileType
        if string == '-':
            msg = (
                "argument type PathType does not support '-' for referring "
                "to sys.std{in,out}"
            )
            raise ValueError(msg)

        # all other arguments are used as file names
        path = Path(string)
        if self._must_exist and not path.exists():
            message = 'path must exist: {}'.format(string)
            raise argparse.ArgumentTypeError(message)
        return str(path)

    def __repr__(self):
        arg_str = repr(self._must_exist)
        return '{}({})'.format(type(self).__name__, arg_str)


def io_opts(parser):
    # Logging
    group = parser.add_argument_group('I/O')
    group.add_argument(
        '--save-config',
        required=False,
        type=PathType(exists=False),
        is_write_out_config_file_arg=False,
        # Setting it to true makes it save and exit
        help='Save parsed configuration and arguments to the specified file',
    )
    group.add_argument(
        '-d', '--debug', action='store_true', help='Output additional messages.'
    )
    group.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='Only output warning and error messages.',
    )


def logging_opts(parser):
    # Logging options
    group = parser.add_argument_group('Logging')
    group.add_argument(
        '--log-interval', type=int, default=100, help='Log every k batches.'
    )
    group.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default='mlruns/',
        help='If using MLflow, logs model parameters, training metrics, and '
        'artifacts (files) to this MLflow server. Uses the localhost by '
        'default.',
    )
    group.add_argument(
        '--experiment-name',
        required=False,
        help='If using MLflow, it will log this run under this experiment '
        'name, which appears as a separate section'
        'in the UI. It will also be used in some messages and files.',
    )
    group.add_argument(
        '--run-name',
        required=False,
        help='If using MLflow, it will log this run under this run '
        'name, which appears as a separate item in the experiment.',
    )
    group.add_argument(
        '--run-uuid',
        required=False,
        help='If specified, MLflow/Default Logger will log metrics and params '
        'under this ID. If it exists, the run status will '
        'change to running. This ID is also used for creating '
        'this run\'s output directory. '
        '(Run ID must be a 32-character hex string)',
    )
    group.add_argument(
        '--output-dir',
        type=str,
        help='Output several files for this run under this directory. '
        'If not specified, a directory under "runs" is created '
        'or reused based on the Run UUID. '
        'Files might also be sent to MLflow depending on the '
        '--mlflow-always-log-artifacts option.',
    )
    group.add_argument(
        '--mlflow-always-log-artifacts',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='If using MLFlow, always log (send) artifacts (files) to MLflow '
        'artifacts URI. By default (false), artifacts are only logged if'
        'MLflow is a remote server (as specified by --mlflow-tracking-uri '
        'option). All generated files are always saved in --output-dir, so it '
        'might be considered redundant to copy them to a local MLflow '
        'server. If this is not the case, set this option to true.',
    )


def general_opts(parser):
    # Data processing options
    group = parser.add_argument_group('random')
    group.add_argument('--seed', type=int, default=42, help='Random seed')

    # Cuda
    group = parser.add_argument_group('gpu')
    group.add_argument(
        '--gpu-id',
        default=None,
        type=int,
        help='Use CUDA on the listed devices',
    )


def save_load_opts(parser):
    group = parser.add_argument_group('save-load')
    group.add_argument(
        '--load-model',
        type=PathType(exists=True),
        help='Directory containing a {} file to be loaded'.format(
            const.MODEL_FILE
        ),
    )
    group.add_argument(
        '--save-data',
        type=str,
        help='Output dir for saving the preprocessed data files.',
    )
    group.add_argument(
        '--load-data',
        type=PathType(exists=True),
        help='Input dir for loading the preprocessed data files.',
    )

    group.add_argument(
        '--load-vocab',
        type=PathType(exists=True),
        help='Directory containing a {} file to be loaded'.format(
            const.VOCAB_FILE
        ),
    )
