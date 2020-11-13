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
import yaml

from kiwi.lib import pretrain

predictor_yaml = """
class_name: Predictor

num_data_workers: 0
batch_size:
    train: 32
    valid: 32

model:
    encoder:
        hidden_size: 400
        rnn_layers: 2
        embeddings:
            source:
                dim: 200
            target:
                dim: 200
        out_embeddings_dim: 200
        share_embeddings: false
        dropout: 0.5
        use_mismatch_features: false

optimizer:
    class_name: adam
    learning_rate: 0.001
    learning_rate_decay: 0.6
    learning_rate_decay_start: 2

data_processing:
    vocab:
        min_frequency: 1
        max_size: 60_000
"""


@pytest.fixture
def predictor_config_dict():
    return yaml.unsafe_load(predictor_yaml)


def test_pretrain_predictor(
    tmp_path, predictor_config_dict, pretrain_config, data_config, extra_big_atol
):
    pretrain_config['run']['output_dir'] = tmp_path
    pretrain_config['data'] = data_config
    pretrain_config['system'] = predictor_config_dict

    train_info = pretrain.pretrain_from_configuration(pretrain_config)

    stats = train_info.best_metrics
    np.testing.assert_allclose(stats['target_PERP'], 838.528486, atol=extra_big_atol)
    np.testing.assert_allclose(
        stats['val_target_PERP'], 501.516467, atol=extra_big_atol
    )

    # Testing predictor with pickled data
    pretrain_config['system']['load'] = train_info.best_model_path

    train_info_from_loaded = pretrain.pretrain_from_configuration(pretrain_config)

    stats = train_info_from_loaded.best_metrics
    np.testing.assert_allclose(
        stats['target_PERP'], 166.4964834955168, atol=extra_big_atol
    )
    np.testing.assert_allclose(
        stats['val_target_PERP'], 333.688725, atol=extra_big_atol
    )


if __name__ == '__main__':  # pragma: no cover

    pytest.main([__file__])  # pragma: no cover
