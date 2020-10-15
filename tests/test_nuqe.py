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
import numpy as np
import pytest

from conftest import check_computation
from kiwi import constants as const
from kiwi.lib import train
from kiwi.lib.utils import save_config_to_file


@pytest.fixture
def nuqe_config_dict(optimizer_config, data_processing_config):
    encoder = dict(
        window_size=3, embeddings=dict(source=dict(dim=50), target=dict(dim=50))
    )
    decoder_side = dict(hidden_sizes=[40, 20, 10, 5], dropout=0.0)
    decoder = dict(target=decoder_side, source=decoder_side)
    outputs = dict(
        word_level=dict(
            target=False,
            gaps=False,
            source=False,
            class_weights=dict(
                target_tags={const.BAD: 3.0},
                gap_tags={const.BAD: 5.0},
                source_tags={const.BAD: 5.0},
            ),
        ),
        sentence_level=dict(hter=False, use_distribution=True, binary=False),
    )

    config = dict(
        class_name='NuQE',
        batch_size=8,
        num_data_workers=0,
        model=dict(encoder=encoder, decoder=decoder, outputs=outputs),
        optimizer=optimizer_config,
        data_processing=data_processing_config,
    )

    return config


@pytest.fixture
def output_target_config(nuqe_config_dict):
    nuqe_config_dict['model']['outputs']['word_level']['target'] = True
    return nuqe_config_dict


@pytest.fixture
def output_source_config(nuqe_config_dict):
    nuqe_config_dict['model']['outputs']['word_level']['source'] = True
    return nuqe_config_dict


@pytest.fixture
def output_gaps_config(nuqe_config_dict):
    nuqe_config_dict['model']['outputs']['word_level']['gaps'] = True
    return nuqe_config_dict


@pytest.fixture
def output_targetgaps_config(nuqe_config_dict):
    nuqe_config_dict['model']['outputs']['word_level']['target'] = True
    nuqe_config_dict['model']['outputs']['word_level']['gaps'] = True
    return nuqe_config_dict


def test_computation_target(
    tmp_path, output_target_config, train_config, data_config, atol
):
    train_config['data'] = data_config
    train_config['system'] = output_target_config
    check_computation(
        train_config,
        tmp_path,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.498354,
        atol=atol,
    )


def test_computation_gaps(
    tmp_path, output_gaps_config, train_config, data_config, atol
):
    train_config['data'] = data_config
    train_config['system'] = output_gaps_config
    check_computation(
        train_config,
        tmp_path,
        output_name=const.GAP_TAGS,
        expected_avg_probs=0.316064,
        atol=atol,
    )


def test_computation_source(
    tmp_path, output_source_config, train_config, data_config, atol
):
    train_config['data'] = data_config
    train_config['system'] = output_source_config
    check_computation(
        train_config,
        tmp_path,
        output_name=const.SOURCE_TAGS,
        expected_avg_probs=0.486522,
        atol=atol,
    )


def test_computation_targetgaps(
    tmp_path, output_targetgaps_config, train_config, data_config, atol
):
    train_config['data'] = data_config
    train_config['system'] = output_targetgaps_config
    check_computation(
        train_config,
        tmp_path,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.507699,
        atol=atol,
    )


def test_api(tmp_path, output_target_config, train_config, data_config, atol):
    from kiwi import train_from_file, load_system

    train_config['data'] = data_config
    train_config['system'] = output_target_config
    train_config['run']['use_mlflow'] = True

    config_file = tmp_path / 'config.yaml'
    save_config_to_file(train.Configuration(**train_config), config_file)

    train_run_info = train_from_file(config_file)

    runner = load_system(train_run_info.best_model_path)

    source = open(data_config['test']['input']['source']).readlines()
    target = open(data_config['test']['input']['target']).readlines()
    alignments = open(data_config['test']['input']['alignments']).readlines()

    predictions = runner.predict(
        source=source,
        target=target,
        alignments=alignments,
        batch_size=train_config['system']['batch_size'],
    )

    target_tags_probabilities = predictions.target_tags_BAD_probabilities
    avg_of_avgs = np.mean(list(map(np.mean, target_tags_probabilities)))
    max_prob = max(map(max, target_tags_probabilities))
    min_prob = min(map(min, target_tags_probabilities))
    np.testing.assert_allclose(avg_of_avgs, 0.498287, atol=atol)
    assert 0 <= min_prob <= avg_of_avgs <= max_prob <= 1

    assert len(predictions.target_tags_labels) == len(target)


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
