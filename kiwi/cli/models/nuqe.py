from distutils.util import strtobool

from kiwi.cli.better_argparse import ModelParser
from kiwi.cli.models.quetch import (
    add_data_flags,
    add_predicting_options,
    add_training_data_file_opts,
    add_vocabulary_opts,
)
from kiwi.models.nuqe import NuQE


def add_model_hyper_params_opts(training_parser):
    group = training_parser.add_argument_group('hyper-parameters')
    group.add_argument(
        '--bad-weight',
        type=float,
        default=3.0,
        help='Relative weight for bad labels.',
    )
    group.add_argument(
        '--window-size', type=int, default=3, help='Sliding window size.'
    )
    group.add_argument(
        '--max-aligned',
        type=int,
        default=5,
        help='Max number of alignments between source and target.',
    )
    group.add_argument(
        '--source-embeddings-size',
        type=int,
        default=50,
        help='Word embedding size for source.',
    )
    group.add_argument(
        '--target-embeddings-size',
        type=int,
        default=50,
        help='Word embedding size for target.',
    )
    group.add_argument(
        '--freeze-embeddings',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Freeze embedding weights during training.',
    )
    group.add_argument(
        '--embeddings-dropout',
        type=float,
        default=0.0,
        help='Dropout rate for embedding layers.',
    )

    group.add_argument(
        '--hidden-sizes',
        type=int,
        nargs='+',
        # action='append',
        default=[400, 200, 100, 50],
        help='List of hidden sizes.',
    )
    group.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout rate for linear layers.',
    )
    group.add_argument(
        '--init-type',
        type=str,
        default='uniform',
        choices=[
            'uniform',
            'normal',
            'constant',
            'glorot_uniform',
            'glorot_normal',
        ],
        help='Distribution type for parameters initialization.',
    )
    group.add_argument(
        '--init-support',
        type=float,
        default=0.1,
        help='Parameters are initialized over uniform distribution with '
        'support (-param_init, param_init). Use 0 to not use '
        'initialization.',
    )

    return group


def add_training_options(training_parser):
    add_training_data_file_opts(training_parser)
    add_data_flags(training_parser)
    add_vocabulary_opts(training_parser)
    add_model_hyper_params_opts(training_parser)


def parser_for_pipeline(pipeline):
    if pipeline == 'train':
        return ModelParser(
            'nuqe',
            'train',
            title=NuQE.title,
            options_fn=add_training_options,
            api_module=NuQE,
        )
    if pipeline == 'predict':
        return ModelParser(
            'nuqe',
            'predict',
            title=NuQE.title,
            options_fn=add_predicting_options,
            api_module=NuQE,
        )

    return None
