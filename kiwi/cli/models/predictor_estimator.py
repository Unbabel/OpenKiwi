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

from kiwi import constants as const
from kiwi.cli.better_argparse import ModelParser
from kiwi.cli.opts import PathType
from kiwi.lib.utils import parse_integer_with_positive_infinity
from kiwi.models.predictor_estimator import Estimator

title = 'Estimator (Predictor-Estimator)'


def _add_training_data_file_opts(parser):
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
        # required=True,
        help='Path to training target file',
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
        '--train-pe',
        type=PathType(exists=True),
        help='Path to file containing post-edited target.',
    )
    group.add_argument(
        '--train-sentence-scores',
        type=PathType(exists=True),
        help='Path to file containing sentence level scores.',
    )

    valid_group = parser.add_argument_group('validation data')

    valid_group.add_argument(
        '--split',
        type=float,
        help='Split Train dataset in case that no validation set is given.',
    )

    valid_group.add_argument(
        '--valid-source',
        type=PathType(exists=True),
        # required=True,
        help='Path to validation source file',
    )
    valid_group.add_argument(
        '--valid-target',
        type=PathType(exists=True),
        # required=True,
        help='Path to validation target file',
    )
    valid_group.add_argument(
        '--valid-alignments',
        type=str,
        # required=True,
        help='Path to valid alignments between source and target.',
    )
    valid_group.add_argument(
        '--valid-source-tags',
        type=PathType(exists=True),
        help='Path to validation label file for source (WMT18 format)',
    )
    valid_group.add_argument(
        '--valid-target-tags',
        type=PathType(exists=True),
        help='Path to validation label file for target',
    )

    valid_group.add_argument(
        '--valid-pe',
        type=PathType(exists=True),
        help='Path to file containing postedited target.',
    )
    valid_group.add_argument(
        '--valid-sentence-scores',
        type=PathType(exists=True),
        help='Path to file containing sentence level scores.',
    )


def _add_predicting_data_file_opts(parser):
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
    return group


def _add_data_flags(parser):
    group = parser.add_argument_group('data processing options')
    group.add_argument(
        '--predict-side',
        type=str,
        default=const.TARGET_TAGS,
        choices=[const.TARGET_TAGS, const.SOURCE_TAGS, const.GAP_TAGS],
        help='Tagset to predict. Leave unchanged for WMT17 format.',
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
        type=parse_integer_with_positive_infinity,
        default=float("inf"),
        help='Maximum source sequence length',
    )
    group.add_argument(
        '--source-min-length',
        type=int,
        default=0,
        help='Truncate source sequence length.',
    )
    group.add_argument(
        '--target-max-length',
        type=parse_integer_with_positive_infinity,
        default=float("inf"),
        help='Maximum target sequence length to keep.',
    )
    group.add_argument(
        '--target-min-length',
        type=int,
        default=0,
        help='Truncate target sequence length.',
    )

    return group


def _add_vocabulary_opts(parser):
    group = parser.add_argument_group(
        'vocabulary options',
        description='Options for loading vocabulary from a previous run. '
        'This is used for e.g. training a source predictor via predict-'
        'inverse: True ; If set, other vocab options are ignored',
    )
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


def _add_data_options(data_parser):
    group = data_parser.add_argument_group(
        'PredEst data',
        description='Predictor Estimator specific data ' 'options. (POSTECH)',
    )

    group.add(
        '--extend-source-vocab',
        type=PathType(exists=True),
        help='Optionally load more data which is used only for vocabulary '
        'creation. Path to additional Data'
        '(Predictor)',
    )
    group.add(
        '--extend-target-vocab',
        type=PathType(exists=True),
        help='Optionally load more data which is used only for vocabulary '
        'creation. Path to additional Data'
        '(Predictor)',
    )


def add_pretraining_options(parser):
    _add_training_data_file_opts(parser)
    _add_data_flags(parser)
    _add_vocabulary_opts(parser)
    _add_data_options(parser)

    group = parser.add_argument_group(
        'predictor training', description='Predictor Estimator (POSTECH)'
    )
    # Only for training
    group.add_argument(
        '--warmup',
        type=int,
        default=0,
        help='Pretrain Predictor for this number of steps.',
    )
    group.add_argument(
        '--rnn-layers-pred', type=int, default=2, help='Layers in Pred RNN'
    )
    group.add_argument(
        '--dropout-pred', type=float, default=0.0, help='Dropout in predictor'
    )
    group.add_argument(
        '--hidden-pred',
        type=int,
        default=100,
        help='Size of hidden layers in LSTM',
    )
    group.add_argument(
        '--out-embeddings-size',
        type=int,
        default=200,
        help='Word Embedding in Output layer',
    )
    group.add_argument(
        '--embedding-sizes',
        type=int,
        default=0,
        help='If set, takes precedence over other embedding params',
    )
    group.add_argument(
        '--share-embeddings',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Tie input and output embeddings for target.',
    )

    group.add_argument(
        '--predict-inverse',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Predict target -> source instead of source -> target.',
    )

    group = parser.add_argument_group(
        'model-embeddings',
        description='Embedding layers size in case pre-trained embeddings '
        'are not used.',
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


def add_training_options(training_parser):
    add_pretraining_options(training_parser)

    group = training_parser.add_argument_group(
        'predictor-estimator training',
        description='Predictor Estimator (POSTECH). These settings are used '
        ' to train the Predictor. They will be ignored if training a '
        ' Predictor-Estimator and the `load-model` flag is set.',
    )
    group.add_argument(
        '--start-stop',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Append start and stop symbols to estimator feature sequence.',
    )
    group.add_argument(
        '--predict-gaps',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Predict Gap Tags. Requires `train-gap-tags`, `valid-'
        'gap-tags` to be set.',
    )
    group.add_argument(
        '--predict-target',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=True,
        help='Predict Target Tags. Requires `train-target-tags`, `valid-'
        'target-tags` to be set.',
    )
    group.add_argument(
        '--predict-source',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Predict Source Tags. Requires `train-source-tags`, `valid-'
        'source-tags` to be set.',
    )
    group.add_argument(
        '--load-pred-source',
        type=PathType(exists=True),
        help='If set, model architecture and vocabulary parameters are '
        'ignored. Load pretrained predictor tgt->src.',
    )
    group.add_argument(
        '--load-pred-target',
        type=PathType(exists=True),
        help='If set, model architecture and vocabulary parameters are '
        'ignored. Load pretrained predictor src->tgt.',
    )

    group.add_argument(
        '--rnn-layers-est', type=int, default=2, help='Layers in Estimator RNN'
    )
    group.add_argument(
        '--dropout-est', type=float, default=0.0, help='Dropout in estimator'
    )
    group.add_argument(
        '--hidden-est',
        type=int,
        default=100,
        help='Size of hidden layers in LSTM',
    )
    group.add_argument(
        '--mlp-est',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help="""Pass the Estimator input through a linear layer
        reducing dimensionality before RNN.""",
    )
    group.add_argument(
        '--sentence-level',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help="""Predict Sentence Level Scores.
        Requires setting `train-sentence-scores, valid-sentence-scores`""",
    )
    group.add_argument(
        '--sentence-ll',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help="""Use probabilistic Loss for sentence scores instead of
        squared error. If set, the model will output mean and variance of
        a truncated Gaussian distribution over the interval [0, 1], and use
        the NLL of ground truth `hter` as the loss.
        This seems to improve performance, and gives you uncertainty estimates
        for sentence level predictions as a byproduct.
        If `sentence-level == False`, this is without effect.
        """,
    )
    group.add_argument(
        '--sentence-ll-predict-mean',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help="""If `sentence-ll == True`, by default the prediction for `hter`
        will be the mean of the Guassian /before truncation/. After truncation,
        this will be the mode of the distribution, but not the mean as
        truncated Gaussian is skewed to one side. set this to `True` to use
        the True mean after truncation for prediction.
        """,
    )
    group.add_argument(
        '--use-probs',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Predict scores as product/sum of word level probs',
    )

    group.add_argument(
        '--binary-level',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help="""Predict binary sentence labels indicating `hter == 0.0`
        Requires setting `train-sentence-scores`, `valid-sentence-scores`""",
    )
    group.add_argument(
        '--token-level',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help="""Continue training the predictor on the postedited text.
        If set, will do an additional forward pass through the predictor
        Using the SRC, PE pair and add the `Predictor` loss for the tokens
        in the postedited text PE. Recommended if you have access to PE.
        Requires setting `train-pe`, `valid-pe`""",
    )
    group.add_argument(
        '--target-bad-weight',
        type=float,
        default=3.0,
        help='Relative weight for target bad labels.',
    )
    group.add_argument(
        '--gaps-bad-weight',
        type=float,
        default=3.0,
        help='Relative weight for gaps bad labels.',
    )
    group.add_argument(
        '--source-bad-weight',
        type=float,
        default=3.0,
        help='Relative weight for source bad labels.',
    )


def add_predicting_options(predicting_parser):
    _add_predicting_data_file_opts(predicting_parser)
    group = predicting_parser.add_argument_group(
        'predictor-estimator Prediction',
        description='Predictor Estimator (POSTECH)',
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
        '--sentence-level',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Predict Sentence Level Scores',
    )
    group.add_argument(
        '--binary-level',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Predict binary sentence labels',
    )
    group.add_argument(
        '--valid-batch-size', type=int, default=32, help='Batch Size'
    )
    group.add_argument(
        '--predict-inverse',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
        help='Predict target -> source instead of source -> target.',
    )


def parser_for_pipeline(pipeline):
    if pipeline == 'train':
        return ModelParser(
            'estimator',
            'train',
            title=Estimator.title,
            options_fn=add_training_options,
            api_module=Estimator,
        )
    if pipeline == 'predict':
        return ModelParser(
            'estimator',
            'predict',
            title=Estimator.title,
            options_fn=add_predicting_options,
            api_module=Estimator,
        )

    return None
