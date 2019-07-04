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
from distutils.util import strtobool

from kiwi.cli.better_argparse import PipelineParser
from kiwi.cli.models import linear, nuqe, predictor, predictor_estimator, quetch
from kiwi.lib import train

logger = logging.getLogger(__name__)


def train_opts(parser):
    # Training loop options
    group = parser.add_argument_group('training')

    group.add_argument(
        '--epochs', type=int, default=50, help='Number of epochs for training.'
    )
    group.add_argument(
        '--train-batch-size',
        type=int,
        default=64,
        help='Maximum batch size for training.',
    )
    group.add_argument(
        '--valid-batch-size',
        type=int,
        default=64,
        help='Maximum batch size for evaluating.',
    )

    # Optimization options
    group = parser.add_argument_group('training-optimization')
    group.add_argument(
        '--optimizer',
        default='adam',
        choices=['sgd', 'adagrad', 'adadelta', 'adam', 'sparseadam'],
        help='Optimization method.',
    )
    group.add_argument(
        '--learning-rate',
        type=float,
        default=1.0,
        help='Starting learning rate. '
        'Recommended settings: sgd = 1, adagrad = 0.1, '
        'adadelta = 1, adam = 0.001',
    )
    group.add_argument(
        '--learning-rate-decay',
        type=float,
        default=1.0,
        help='Decay learning rate by this factor. ',
    )
    group.add_argument(
        '--learning-rate-decay-start',
        type=int,
        default=0,
        help='Start decay after this epoch.',
    )

    # Saving and resuming options
    group = parser.add_argument_group('training-save-load')
    group.add_argument(
        '--checkpoint-validation-steps',
        type=int,
        default=0,
        help='Perform validation every X training batches. Saves model'
        ' if `checkpoint-save` is true.',
    )
    group.add_argument(
        '--checkpoint-save',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=True,
        help='Save a training snapshot when validation is run. If false '
        'it will never save the model.',
    )
    group.add_argument(
        '--checkpoint-keep-only-best',
        type=int,
        default=1,
        help='Keep only n best models according to main metric (F1Mult '
        'by default); 0 will keep all.',
    )
    group.add_argument(
        '--checkpoint-early-stop-patience',
        type=int,
        default=0,
        help='Stop training if evaluation metrics do not improve after X '
        'validations; 0 disables this.',
    )
    group.add_argument(
        '--resume',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Resume training a previous run. '
        'If --output-dir is not none, Kiwi will load from a checkpoint folder '
        'in that location. If --output-dir is not specified, '
        'then the --run-uuid option must be set. Files are then searched '
        'under the "runs" directory. If not found, they are '
        'downloaded from the MLflow server '
        '(check the --mlflow-tracking-uri option).',
    )


def build_parser():
    return PipelineParser(
        name='train',
        model_parsers=[
            nuqe.parser_for_pipeline('train'),
            predictor_estimator.parser_for_pipeline('train'),
            predictor.parser_for_pipeline('train'),
            quetch.parser_for_pipeline('train'),
            linear.parser_for_pipeline('train'),
        ],
        options_fn=train_opts,
    )


def main(argv=None):
    parser = build_parser()
    options = parser.parse(args=argv)
    train.train_from_options(options)


if __name__ == '__main__':  # pragma: no cover
    main()  # pragma: no cover
