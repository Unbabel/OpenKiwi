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
from pathlib import Path

import pytest

from kiwi.cli import arguments_to_configuration


@pytest.fixture
def config_path(tmp_path):
    return tmp_path / 'config.yaml'


@pytest.fixture
def config():
    return """
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

    """

def test_arguments_to_configuration(config, config_path):

    config_path.write_text(config)
    config_dict = arguments_to_configuration({'CONFIG_FILE': config_path, 'OVERWRITES': ['class_name=XLMR']})

    assert 'num_data_workers' in config_dict
    assert 'model' in config_dict
    assert 'freeze' in config_dict['model']['encoder']
    assert config_dict['class_name'] == 'XLMR'
