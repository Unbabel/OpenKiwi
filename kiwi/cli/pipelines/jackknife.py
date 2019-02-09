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
