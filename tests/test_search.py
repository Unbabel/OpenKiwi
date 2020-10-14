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
from pydantic import ValidationError

from kiwi.lib import search
from kiwi.lib.utils import save_config_to_file


@pytest.fixture
def search_output():
    return [
        'parallel_coordinate.html',
        'optimization_history.html',
        'search_config.yaml',
        'study.pkl',
        'output.log',
    ]


@pytest.fixture
def xlmr_config_dict():
    return yaml.unsafe_load(
        """
        class_name: XLMRoberta

        num_data_workers: 0
        batch_size:
            train: 16
            valid: 16

        model:
            encoder:
                model_name: xlm-roberta-base
                interleave_input: false
                freeze: false
                encode_source: false
                pooling: mixed

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
    )


@pytest.fixture
def search_config(base_search_config, output_target_config, data_config):
    base_search_config['base_config']['debug'] = True
    base_search_config['base_config']['data'] = data_config
    base_search_config['base_config']['system'] = output_target_config
    return base_search_config


@pytest.fixture
def xlmr_search_config(base_search_config, xlmr_config_dict, data_config):
    base_search_config['base_config']['debug'] = True
    base_search_config['base_config']['data'] = data_config
    base_search_config['base_config']['system'] = xlmr_config_dict
    return base_search_config


def test_config_validation(tmp_path, search_config):

    output_dir = tmp_path / 'config_validation'
    search_config['directory'] = output_dir

    # These values cannot both be set to True
    search_config['options']['search_hter'] = True
    search_config['options']['search_word_level'] = True
    with pytest.raises(ValidationError):
        search.Configuration(**search_config)


def test_api_nuqe(tmp_path, search_config, search_output):

    from kiwi.lib.search import search_from_file

    output_dir = tmp_path / 'search'
    search_config['directory'] = output_dir

    config_file = tmp_path / 'config.yaml'

    search_config['base_config']['trainer']['gradient_accumulation_steps'] = 2
    search_config['base_config']['trainer']['main_metric'] = 'target_tags_MCC'
    search_config['base_config']['system']['model']['outputs']['word_level'][
        'gaps'
    ] = True
    search_config['base_config']['system']['model']['outputs']['word_level'][
        'source'
    ] = True
    save_config_to_file(search.Configuration(**search_config), config_file)

    # Check a first run
    search_from_file(config_file)
    assert [file.name for file in output_dir.glob('*')] == ['0']
    assert set(file.name for file in (output_dir / '0').glob('*')) == set(search_output)

    # Check a second run with different settings
    search_config['options']['search_method'] = 'multivariate_tpe'
    search_config['options']['search_hter'] = False
    search_config['options']['learning_rate'] = None
    search_config['options']['dropout'] = None
    search_config['options']['search_word_level'] = True
    search_config['options']['search_hter'] = False
    search_config['options']['class_weights'] = None
    search_config['base_config']['trainer']['gradient_accumulation_steps'] = 1
    save_config_to_file(search.Configuration(**search_config), config_file)
    search_from_file(config_file)
    assert set([file.name for file in output_dir.glob('*')]) == set(['0', '1'])

    # Check a third time to test the folder backup and load a study
    #   Create a folder that will clash with the next folder name...
    (output_dir / '3').mkdir(parents=True)
    #   ...and oad the previous study.
    search_config['load_study'] = output_dir / '0' / 'study.pkl'
    save_config_to_file(search.Configuration(**search_config), config_file)
    search_from_file(config_file)
    folders = [file.name for file in output_dir.glob('*')]
    assert '0' in folders
    assert '1' in folders
    assert '3' in folders
    # Check backup logic
    assert sum(folder.startswith('3_backup_') for folder in folders) == 1


def test_api_xlmr(tmp_path, xlmr_search_config, search_output):

    from kiwi.lib.search import search_from_file

    output_dir = tmp_path / 'search_xlmr'
    xlmr_search_config['directory'] = output_dir

    config_file = tmp_path / 'xlmr_config.yaml'

    xlmr_search_config['num_trials'] = 1
    xlmr_search_config['options']['warmup_steps'] = [1, 2]
    xlmr_search_config['options']['freeze_epochs'] = [1, 2]
    xlmr_search_config['options']['hidden_size'] = [10, 20]
    xlmr_search_config['options']['bottleneck_size'] = [10, 20]
    xlmr_search_config['options']['search_mlp'] = True
    save_config_to_file(search.Configuration(**xlmr_search_config), config_file)

    # Will complain becaues the metric is not set
    with pytest.raises(ValueError):
        search_from_file(config_file)
    assert [file.name for file in output_dir.glob('*')] == ['0']

    # This will run
    xlmr_search_config['base_config']['trainer']['main_metric'] = ['WMT19_MCC']
    save_config_to_file(search.Configuration(**xlmr_search_config), config_file)
    search_from_file(config_file)

    assert set([file.name for file in output_dir.glob('*')]) == set(['0', '1'])
    assert set(file.name for file in (output_dir / '1').glob('*')) == set(search_output)


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
