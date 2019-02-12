import configargparse

from kiwi import __version__
from kiwi.cli.pipelines import jackknife, predict, train


def build_parser():
    global parser
    parser = configargparse.get_argument_parser(
        name='main',
        prog='kiwi',
        description='Quality Estimation toolkit',
        add_help=True,
    )
    # prog=None, usage=None, description=None, epilog=None, parents=[],
    # formatter_class=argparse.HelpFormatter, prefix_chars='-',
    # fromfile_prefix_chars=None,
    # argument_default=None, conflict_handler='error', add_help=True
    parser.add_argument('--version', action='version', version=__version__)
    # subparsers = parser.add_subparsers()
    # help_parser = subparsers.add_parser('help', add_help=False)
    # help_parser.add_argument('-h', '--help', action='store_true')
    # pipeline_parser = subparsers.add_parser('pipeline')
    # pipeline_parser.add_argument(
    #     dest='pipeline',
    #     # title='Pipelines',
    #     # description="Use 'kiwi <pipeline> (-h | --help)' to check it out.",
    #     help='Run "kiwi <pipeline> (-h | --help) for more information.',
    #     choices=['train', 'predict', 'search', 'jackknife', 'preprocess',
    #              'pretrain'])
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
        'search',
        # parents=[search.parser],
        add_help=False,
        help='Search training hyperparameters for a QE model',
    )
    subparsers.add_parser(
        'jackknife',
        # parents=[jackknife.parser],
        add_help=False,
        help='Jackknife training data with model',
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


if __name__ == '__main__':  # pragma: no cover
    cli()  # pragma: no cover
