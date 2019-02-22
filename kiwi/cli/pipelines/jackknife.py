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

import logging

from kiwi.cli.better_argparse import HyperPipelineParser
from kiwi.cli.opts import PathType
from kiwi.cli.pipelines import train
from kiwi.lib import jackknife

logger = logging.getLogger(__name__)


def jackknife_opts(parser):
    # Training loop options
    group = parser.add_argument_group('jackknifing')

    group.add(
        '--splits',
        required=False,
        type=int,
        default=5,
        help='Jackknife with X folds.',
    )
    group.add(
        '--train-config',
        required=False,
        type=PathType(exists=True),
        help='Path to config file with model parameters.',
    )


def build_parser():
    return HyperPipelineParser(
        name='jackknife',
        pipeline_parser=train.build_parser(),
        pipeline_config_key='train-config',
        options_fn=jackknife_opts,
    )


def main(argv=None):
    parser = build_parser()
    options = parser.parse(args=argv)
    jackknife.run_from_options(options)


if __name__ == '__main__':  # pragma: no cover
    main()  # pragma: no cover
