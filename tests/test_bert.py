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

import pytest
import yaml
from transformers import BertConfig, BertModel
from transformers.tokenization_bert import VOCAB_FILES_NAMES

from conftest import check_computation
from kiwi import constants as const

bert_yaml = """
class_name: Bert

num_data_workers: 0
batch_size:
    train: 32
    valid: 32

model:
    encoder:
        model_name: None
        interleave_input: false
        freeze: false
        use_mismatch_features: false
        use_predictor_features: false
        encode_source: false

    decoder:
        hidden_size: 16
        dropout: 0.0

    outputs:
        word_level:
            target: true
            gaps: true
            source: true
            class_weights:
                target_tags:
                    BAD: 3.0
                gap_tags:
                    BAD: 5.0
                source_tags:
                    BAD: 3.0
        sentence_level:
            hter: true
            use_distribution: true
            binary: false
        sentence_loss_weight: 1

    tlm_outputs:
        fine_tune: false

optimizer:
    class_name: noam
    learning_rate: 0.00001
    warmup_steps: 1000
    training_steps: 12000

data_processing:
    share_input_fields_encoders: true

"""


@pytest.fixture
def bert_config_dict():
    return yaml.unsafe_load(bert_yaml)


@pytest.fixture(scope='session')
def bert_model_dir(model_dir):
    return model_dir.joinpath('bert/')


@pytest.fixture(scope='function')
def bert_model():
    config = BertConfig(
        vocab_size=107,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        is_decoder=False,
        initializer_range=0.02,
    )
    return BertModel(config=config)


def test_computation_target(
    tmp_path,
    bert_model,
    bert_model_dir,
    bert_config_dict,
    train_config,
    data_config,
    big_atol,
):
    train_config['run']['output_dir'] = tmp_path
    train_config['data'] = data_config
    train_config['system'] = bert_config_dict

    shutil.copy2(bert_model_dir / VOCAB_FILES_NAMES['vocab_file'], tmp_path)
    bert_model.save_pretrained(tmp_path)
    train_config['system']['model']['encoder']['model_name'] = str(tmp_path)

    check_computation(
        train_config,
        tmp_path,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.550805,
        atol=big_atol,
    )


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
