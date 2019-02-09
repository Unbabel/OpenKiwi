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
        help='Perform validation every X training batches.',
    )
    group.add_argument(
        '--checkpoint-save',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Save a training snapshot when validation is run.',
    )
    group.add_argument(
        '--checkpoint-keep-only-best',
        type=int,
        default=0,
        help='Keep only this number of saved snapshots; 0 will keep all.',
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
        'The --run-uuid (and possibly --experiment-name) '
        'option must be specified. Files are then searched '
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
