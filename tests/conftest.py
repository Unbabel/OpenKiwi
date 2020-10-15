#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2020 Unbabel <openkiwi@unbabel.com>
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
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from kiwi import constants as const
from kiwi.lib import predict, train
from kiwi.lib.utils import configure_seed


@pytest.fixture(autouse=True)
def ignore_warnings():
    """Ignore expected warnings."""
    warnings.simplefilter('once', DeprecationWarning)

    # PyTorch-Lightning warning
    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        message='The dataloader, .*, does not have many workers which may be a '
        'bottleneck. Consider increasing the value of the `num_workers` '
        'argument` in the `DataLoader` init to improve performance.',
    )

    # Warning due to AdamW optimizer (HF-transformers) with PyTorch 1.5
    # https://github.com/huggingface/transformers/issues/3961
    warnings.filterwarnings(
        'ignore', category=UserWarning, message='This overload of add_ is deprecated',
    )


@pytest.fixture(scope='function', autouse=True)
def set_seed():
    configure_seed(42)


def setup_function():
    configure_seed(42)


@pytest.fixture(scope='function')
def temp_output_dir():
    output_dir = tempfile.mkdtemp()
    yield Path(output_dir)
    shutil.rmtree(output_dir)


@pytest.fixture(scope='session')
def atol():
    return 1e-3


@pytest.fixture(scope='session')
def big_atol():
    return 0.1


@pytest.fixture(scope='session')
def extra_big_atol():
    return 1.0


@pytest.fixture(scope='session')
def base_dir():
    return Path(__file__).absolute().parent.joinpath('./toy-data/')


@pytest.fixture(scope='session')
def dir_18(base_dir):
    return base_dir.joinpath('WMT18/word_level/en_de.nmt/')


@pytest.fixture(scope='session')
def model_dir(base_dir):
    return base_dir.joinpath('models/')


@pytest.fixture
def data_processing_config():
    config = dict(
        share_input_fields_encoders=False,
        vocab=dict(
            min_frequency=1,
            max_size=100_000,
            # TODO: add these when feature is added.
            # source_max_length=50,
            # source_min_length=1,
            # target_max_length=50,
            # target_min_length=1,
            add_embeddings_vocab=False,
            keep_rare_words_with_embeddings=True,
        ),
    )
    return config


def make_data_config(directory: Path):
    config = {}
    for set_name, file_name in [('train', 'train'), ('valid', 'dev'), ('test', 'dev')]:
        if set_name is not 'test':
            split_config = dict(
                input=dict(
                    source=directory / f'{file_name}.src',
                    target=directory / f'{file_name}.mt',
                    post_edit=directory / f'{file_name}.pe',
                    alignments=directory / f'{file_name}.src-mt.alignments',
                ),
                output=dict(
                    source_tags=directory / f'{file_name}.src_tags',
                    target_tags=directory / f'{file_name}.tags',
                    sentence_scores=directory / f'{file_name}.hter',
                ),
            )
        else:
            # test should have no output
            split_config = dict(
                input=dict(
                    source=directory / f'{file_name}.src',
                    target=directory / f'{file_name}.mt',
                    post_edit=directory / f'{file_name}.pe',
                    alignments=directory / f'{file_name}.src-mt.alignments',
                )
            )
        config[set_name] = split_config

    return config


@pytest.fixture
def data_config(dir_18):
    return make_data_config(dir_18)


@pytest.fixture(scope='function')
def run_config():
    config = dict(seed=42)
    configure_seed(config['seed'])
    return config


@pytest.fixture(scope='function')
def pretrain_config(run_config: dict):
    config = dict(
        run=run_config,
        trainer=dict(
            gpus=0,  # use CPU
            precision=32,
            amp_level='O0',
            epochs=1,
            checkpoint=dict(validation_steps=0.5, save_top_k=1, early_stop_patience=0),
        ),
    )
    return config


@pytest.fixture(scope='function')
def train_config(run_config: dict):
    config = dict(
        run=run_config,
        quiet=True,
        trainer=dict(
            gpus=0,  # use CPU
            epochs=2,
            precision=32,
            amp_level='O0',
            checkpoint=dict(validation_steps=1.0, save_top_k=1, early_stop_patience=0),
        ),
    )
    return config


@pytest.fixture
def weights_config():
    return {
        const.TARGET_TAGS: {const.BAD: 5.0},
        const.GAP_TAGS: {const.BAD: 5.0},
        const.SOURCE_TAGS: {const.BAD: 5.0},
    }


@pytest.fixture(scope='function')
def predest_opts(weights_config, data_opts_17):
    opts = merge_namespaces(weights_config, data_opts_17)

    opts.load_pred_source = None
    opts.load_pred_target = None

    opts.extend_source_vocab = None
    opts.extend_target_vocab = None

    # network
    # opts.embedding_sizes = 0
    opts.share_embeddings = False
    opts.embeddings_size = {const.SOURCE: 25, const.TARGET: 25}

    opts.output_size = None
    opts.dropout = 0.0

    # predest
    opts.warmup = 0
    opts.rnn_layers_pred = 2
    opts.rnn_layers_est = 1
    opts.dropout_est = 0.0
    opts.dropout_pred = 0.0
    opts.hidden_pred = 50
    opts.hidden_est = 25
    opts.mlp_est = False
    opts.out_embeddings_size = 25
    opts.start_stop = False

    # multilevel
    opts.predict_inverse = False
    opts.fine_tune = False
    opts.sentence_ll = False
    opts.use_probs = False
    opts.binary_level = False

    return opts


@pytest.fixture(scope='function')
def predictor_opts(data_opts_17, predest_opts):
    return predest_opts


@pytest.fixture(scope='function')
def transformer_predictor_opts(data_opts_17, predest_opts):
    # for transformer feature embedder
    predest_opts.attention_dropout = 0.0
    predest_opts.nb_heads = 2
    predest_opts.nb_layers = 3
    predest_opts.ff_hidden_size = 50

    # for transformer estimator
    predest_opts.inner_layer_type = 'conv'
    predest_opts.freeze_predictor = False

    return predest_opts


@pytest.fixture
def optimizer_config():
    return dict(
        class_name='adam',
        learning_rate=0.001,
        learning_rate_decay=1.0,
        learning_rate_decay_start=0,
    )


def check_computation(config, output_dir, output_name, expected_avg_probs, atol):
    # Testing training
    config['run']['output_dir'] = output_dir
    training_config = train.Configuration(**config)
    train_info = train.run(training_config)

    # Testing prediction
    # runner = load_model(config.system.load_model)
    load_model = train_info.best_model_path
    predicting_config = predict.Configuration(
        run=dict(seed=42, output_dir=output_dir),
        data=training_config.data,
        system=dict(load=load_model),
        use_gpu=False,
    )

    predictions, metrics = predict.run(predicting_config, output_dir)
    assert metrics is None, 'assuming to run predictions on the test set, so no metrics'

    predictions = predictions[output_name]
    avg_of_avgs = np.mean(list(map(np.mean, predictions)))
    max_prob = max(map(max, predictions))
    min_prob = min(map(min, predictions))
    np.testing.assert_allclose(avg_of_avgs, expected_avg_probs, atol=atol)
    assert 0 <= min_prob <= avg_of_avgs <= max_prob <= 1
