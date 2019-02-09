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

from kiwi.cli.better_argparse import ModelParser
from kiwi.cli.opts import PathType
from kiwi.models.linear_word_qe_classifier import LinearWordQEClassifier

logger = logging.getLogger(__name__)

title = 'linear'


def _add_vocabulary_opts(parser):
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


def add_training_data_file_opts(parser):
    # Data options
    group = parser.add_argument_group('data')

    group.add_argument(
        '--train-source',
        type=PathType(exists=True),
        help='Path to training source file',
    )
    group.add_argument(
        '--train-target',
        type=PathType(exists=True),
        help='Path to training target file',
    )
    group.add_argument(
        '--train-alignments',
        type=str,
        help='Path to train alignments between source and target.',
    )
    group.add_argument(
        '--train-source-tags',
        type=PathType(exists=True),
        help='Path to validation label file for source (WMT18 format)',
    )
    group.add_argument(
        '--train-target-tags',
        type=PathType(exists=True),
        help='Path to validation label file for target',
    )

    group.add_argument(
        '--train-source-pos',
        type=PathType(exists=True),
        help='Path to training PoS tags file for source',
    )
    group.add_argument(
        '--train-target-pos',
        type=PathType(exists=True),
        help='Path to training PoS tags file for target',
    )
    group.add_argument(
        '--train-target-parse',
        type=PathType(exists=True),
        help='Path to training dependency parsing file for target (tabular '
        'format)',
    )
    group.add_argument(
        '--train-target-ngram',
        type=PathType(exists=True),
        help='Path to training highest order ngram file for target (tabular '
        'format)',
    )
    group.add_argument(
        '--train-target-stacked',
        type=PathType(exists=True),
        help='Path to training stacked predictions file for target (tabular '
        'format)',
    )

    group = parser.add_argument_group('validation data')

    group.add_argument(
        '--valid-source',
        type=PathType(exists=True),
        # required=True,
        help='Path to validation source file',
    )
    group.add_argument(
        '--valid-target',
        type=PathType(exists=True),
        # required=True,
        help='Path to validation target file',
    )
    group.add_argument(
        '--valid-alignments',
        type=str,
        # required=True,
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
    group.add_argument(
        '--valid-target-parse',
        type=PathType(exists=True),
        help='Path to validation dependency parsing file for target (tabular '
        'format)',
    )
    group.add_argument(
        '--valid-target-ngram',
        type=PathType(exists=True),
        help='Path to validation highest order ngram file for target (tabular '
        'format)',
    )
    group.add_argument(
        '--valid-target-stacked',
        type=PathType(exists=True),
        help='Path to validation stacked predictions file for target (tabular '
        'format)',
    )


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
    group.add_argument(
        '--test-alignments',
        type=PathType(exists=True),
        help='Path to test alignments between source and target.',
    )
    group.add_argument(
        '--test-source-pos',
        type=PathType(exists=True),
        help='Path to training PoS tags file for source',
    )
    group.add_argument(
        '--test-target-pos',
        type=PathType(exists=True),
        help='Path to training PoS tags file for target',
    )
    group.add_argument(
        '--test-target-parse',
        type=PathType(exists=True),
        help='Path to test dependency parsing file for target (tabular format)',
    )
    group.add_argument(
        '--test-target-ngram',
        type=PathType(exists=True),
        help='Path to test highest order ngram file for target (tabular '
        'format)',
    )  # noqa
    group.add_argument(
        '--test-target-stacked',
        type=PathType(exists=True),
        help='Path to test stacked predictions file for target (tabular '
        'format)',
    )  # noqa

    return group


def _add_output_options(group):
    # Other options (used both at training and test time).
    group.add_argument(
        '--evaluation-metric',
        type=str,
        default='f1_mult',
        help='Evaluation metric (f1_mult or f1_bad).',
    )


def add_training_options(training_parser):
    add_training_data_file_opts(training_parser)
    _add_vocabulary_opts(training_parser)

    group = training_parser.add_argument_group(
        'linear', description='Linear Quality Estimation'
    )

    # Model options (training time).
    group.add_argument(
        '--use-basic-features-only',
        type=int,
        default=0,
        help='1 for using only basic features (words).',
    )
    group.add_argument(
        '--use-bigrams',
        type=int,
        default=1,
        help='1 for using bigram features (i.e. a CRF-like model).',
    )
    group.add_argument(
        '--use-simple-bigram-features',
        type=int,
        default=0,
        help='1 for using only label indicators as bigram features.',
    )

    # Training options.
    group.add_argument(
        '--training-algorithm',
        type=str,
        default='svm_mira',
        help='Algorithm for training the model (svm_mira, svm_sgd, '
        'perceptron).',
    )
    group.add_argument(
        '--regularization-constant',
        type=float,
        default=0.001,
        help='L2 regularization constant.',
    )
    group.add_argument(
        '--cost-false-positives',
        type=float,
        default=0.2,
        help='Cost for false positives (svm_mira and svm_sgd only).',
    )
    group.add_argument(
        '--cost-false-negatives',
        type=float,
        default=0.8,
        help='Cost for false negatives (svm_mira and svm_sgd only).',
    )

    _add_output_options(group)


def add_predicting_options(predicting_parser):
    add_predicting_data_file_opts(predicting_parser)
    _add_output_options(predicting_parser)


def parser_for_pipeline(pipeline):
    if pipeline == 'train':
        return ModelParser(
            'linear',
            'train',
            title=LinearWordQEClassifier.title,
            options_fn=add_training_options,
            api_module=LinearWordQEClassifier,
        )
    if pipeline == 'predict':
        return ModelParser(
            'linear',
            'predict',
            title=LinearWordQEClassifier.title,
            options_fn=add_predicting_options,
            api_module=LinearWordQEClassifier,
        )

    return None
