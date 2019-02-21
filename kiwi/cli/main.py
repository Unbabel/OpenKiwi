import configargparse

from kiwi import __version__
from kiwi.cli.pipelines import evaluate, jackknife, predict, train


def build_parser():
    global parser
    parser = configargparse.get_argument_parser(
        name='main',
        prog='kiwi',
        description='Quality Estimation toolkit',
        add_help=True,
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
    #subparsers.add_parser(
    #    'search',
    #    # parents=[search.parser],
    #    add_help=False,
    #    help='Search training hyperparameters for a QE model',
    #)
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
