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

from distutils.util import strtobool

from kiwi.cli.better_argparse import ModelParser
from kiwi.cli.opts import PathType
from kiwi.models.quetch import QUETCH


def add_training_data_file_opts(parser):
    # Data options
    group = parser.add_argument_group('data')

    group.add_argument(
        '--train-source',
        type=PathType(exists=True),
        required=True,
        help='Path to training source file',
    )
    group.add_argument(
        '--train-target',
        type=PathType(exists=True),
        required=True,
        help='Path to training target file',
    )
    group.add_argument(
        '--train-alignments',
        type=str,
        required=True,
        help='Path to train alignments between source and target.',
    )
    group.add_argument(
        '--train-source-tags',
        type=PathType(exists=True),
        help='Path to training label file for source (WMT18 format)',
    )
    group.add_argument(
        '--train-target-tags',
        type=PathType(exists=True),
        help='Path to training label file for target',
    )

    group.add_argument(
        '--valid-source',
        type=PathType(exists=True),
        required=True,
        help='Path to validation source file',
    )
    group.add_argument(
        '--valid-target',
        type=PathType(exists=True),
        required=True,
        help='Path to validation target file',
    )
    group.add_argument(
        '--valid-alignments',
        type=str,
        required=True,
        help='Path to valid alignments between source and target.',
    )
    group.add_argument(
        '--valid-source-tags',
        type=PathType(exists=True),
        help='Path to validation label file for source (WMT18 format)',
    )
    group.add_argument(
        '--valid-target-tags',
        type=PathType(exists=True),
        help='Path to validation label file for target',
    )

    group.add_argument(
        '--valid-source-pos',
        type=PathType(exists=True),
        help='Path to training PoS tags file for source',
    )
    group.add_argument(
        '--valid-target-pos',
        type=PathType(exists=True),
        help='Path to training PoS tags file for target',
    )

    return group


def add_predicting_data_file_opts(parser):
    # Data options
    group = parser.add_argument_group('data')

    group.add_argument(
        '--test-source',
        type=PathType(exists=True),
        required=True,
        help='Path to validation source file',
    )
    group.add_argument(
        '--test-target',
        type=PathType(exists=True),
        required=True,
        help='Path to validation target file',
    )
    group.add(
        '--test-alignments',
        type=PathType(exists=True),
        required=True,
        help='Path to test alignments between source and target.',
    )

    return group


def add_data_flags(parser):
    group = parser.add_argument_group('data processing options')
    group.add_argument(
        '--predict-target',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=True,
        help='Predict Target Tags. Leave unchanged for WMT17 format',
    )
    group.add_argument(
        '--predict-gaps',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Predict Gap Tags.',
    )
    group.add_argument(
        '--predict-source',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Predict Source Tags.',
    )
    group.add_argument(
        '--wmt18-format',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Read target tags in WMT18 format.',
    )

    group.add_argument(
        '--source-max-length',
        type=int,
        default=float("inf"),
        help='Maximum source sequence length',
    )
    group.add_argument(
        '--source-min-length',
        type=int,
        default=1,
        help='Truncate source sequence length.',
    )
    group.add_argument(
        '--target-max-length',
        type=int,
        default=float("inf"),
        help='Maximum target sequence length to keep.',
    )
    group.add_argument(
        '--target-min-length',
        type=int,
        default=1,
        help='Truncate target sequence length.',
    )

    return group


def add_vocabulary_opts(parser):
    group = parser.add_argument_group('vocabulary options')
    group.add_argument(
        '--source-vocab-size',
        type=int,
        default=None,
        help='Size of the source vocabulary.',
    )
    group.add_argument(
        '--target-vocab-size',
        type=int,
        default=None,
        help='Size of the target vocabulary.',
    )
    group.add_argument(
        '--source-vocab-min-frequency',
        type=int,
        default=1,
        help='Min word frequency for source vocabulary.',
    )
    group.add_argument(
        '--target-vocab-min-frequency',
        type=int,
        default=1,
        help='Min word frequency for target vocabulary.',
    )

    group.add_argument(
        '--keep-rare-words-with-embeddings',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Keep words that occur less then min-frequency but '
        'are in embeddings vocabulary.',
    )
    group.add_argument(
        '--add-embeddings-vocab',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Add words from embeddings vocabulary to source/target '
        'vocabulary.',
    )

    group.add_argument(
        '--embeddings-format',
        type=str,
        default='polyglot',
        choices=['polyglot', 'word2vec', 'fasttext', 'glove', 'text'],
        help='Word embeddings format. '
        'See README for specific formatting instructions.',
    )
    group.add_argument(
        '--embeddings-binary',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Load embeddings stored in binary.',
    )
    group.add_argument(
        '--source-embeddings',
        type=PathType(exists=True),
        help='Path to word embeddings file for source.',
    )
    group.add_argument(
        '--target-embeddings',
        type=PathType(exists=True),
        help='Path to word embeddings file for target.',
    )

    return group


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
        default=[50],
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


def add_predicting_options(predicting_parser):
    add_predicting_data_file_opts(predicting_parser)
    add_data_flags(predicting_parser)


def parser_for_pipeline(pipeline):
    if pipeline == 'train':
        return ModelParser(
            'quetch',
            'train',
            title=QUETCH.title,
            options_fn=add_training_options,
            api_module=QUETCH,
        )
    if pipeline == 'predict':
        return ModelParser(
            'quetch',
            'predict',
            title=QUETCH.title,
            options_fn=add_predicting_options,
            api_module=QUETCH,
        )

    return None
