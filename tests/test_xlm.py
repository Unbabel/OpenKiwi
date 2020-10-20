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
import pytest
import yaml
from transformers import XLMConfig, XLMModel, XLMTokenizer

from conftest import check_computation
from kiwi import constants as const

xlm_yaml = """
class_name: XLM

num_data_workers: 0
batch_size:
    train: 32
    valid: 32

model:
    encoder:
        model_name: None
        interleave_input: false
        freeze: false
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
    class_name: adamw
    learning_rate: 0.00001
    warmup_steps: 0.1
    training_steps: 12000

data_processing:
    share_input_fields_encoders: true

"""


@pytest.fixture
def xlm_config_dict():
    return yaml.unsafe_load(xlm_yaml)


@pytest.fixture(scope='session')
def xlm_model_dir(model_dir):
    return model_dir.joinpath('xlm/')


@pytest.fixture(scope='function')
def xlm_model():
    config = XLMConfig(
        vocab_size=93000,
        emb_dim=32,
        n_layers=5,
        n_heads=4,
        dropout=0.1,
        max_position_embeddings=512,
        lang2id={
            "ar": 0,
            "bg": 1,
            "de": 2,
            "el": 3,
            "en": 4,
            "es": 5,
            "fr": 6,
            "hi": 7,
            "ru": 8,
            "sw": 9,
            "th": 10,
            "tr": 11,
            "ur": 12,
            "vi": 13,
            "zh": 14,
        },
    )
    return XLMModel(config=config)


@pytest.fixture()
def xlm_tokenizer():
    return XLMTokenizer.from_pretrained('xlm-mlm-tlm-xnli15-1024')


def test_computation_target(
    tmp_path,
    xlm_model,
    xlm_tokenizer,
    xlm_model_dir,
    xlm_config_dict,
    train_config,
    data_config,
    big_atol,
):
    train_config['run']['output_dir'] = tmp_path
    train_config['data'] = data_config
    train_config['system'] = xlm_config_dict

    xlm_model.save_pretrained(tmp_path)
    xlm_tokenizer.save_pretrained(tmp_path)
    train_config['system']['model']['encoder']['model_name'] = str(tmp_path)

    check_computation(
        train_config,
        tmp_path,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.410072,
        atol=big_atol,
    )


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
