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

import configargparse

from kiwi import __copyright__, __version__
from kiwi.cli.pipelines import evaluate, jackknife, predict, train


def build_parser():
    global parser
    parser = configargparse.get_argument_parser(
        name='main',
        prog='kiwi',
        description='Quality Estimation toolkit',
        add_help=True,
        epilog='Copyright {}'.format(__copyright__),
    )
    parser.add_argument('--version', action='version', version=__version__)
    subparsers = parser.add_subparsers(
        title='Pipelines',
        description="Use 'kiwi <pipeline> (-h | --help)' to check it out.",
        help='Available pipelines:',
        dest='pipeline',
    )
    subparsers.required = True
    subparsers.add_parser(
        'train',
        # parents=[train.parser],
        add_help=False,
        help='Train a QE model',
    )
    subparsers.add_parser(
        'predict',
        # parents=[predict.parser],
        add_help=False,
        help='Use a pre-trained model for prediction',
    )
    subparsers.add_parser(
        'jackknife',
        # parents=[jackknife.parser],
        add_help=False,
        help='Jackknife training data with model',
    )
    subparsers.add_parser(
        'evaluate',
        add_help=False,
        help='Evaluate a model\'s predictions using popular metrics',
    )
    return parser


def cli():
    options, extra_args = build_parser().parse_known_args()

    if options.pipeline == 'train':
        train.main(extra_args)
    if options.pipeline == 'predict':
        predict.main(extra_args)
    # Meta pipelines
    # if options.pipeline == 'search':
    #     search.main(extra_args)
    if options.pipeline == 'jackknife':
        jackknife.main(extra_args)
    if options.pipeline == 'evaluate':
        evaluate.main(extra_args)


if __name__ == '__main__':  # pragma: no cover
    cli()  # pragma: no cover
