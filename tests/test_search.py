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
        'best_configs'
    ]


@pytest.fixture
def search_config(base_search_config, output_target_config, data_config):
    base_search_config['base_config']['debug'] = True
    base_search_config['base_config']['data'] = data_config
    base_search_config['base_config']['system'] = output_target_config
    return base_search_config


def test_config_validation(tmp_path, search_config):

    output_dir = tmp_path / 'config_validation'
    search_config['directory'] = output_dir

    # These values cannot both be set to True
    search_config['options']['search_hter'] = True
    search_config['options']['search_word_level'] = True
    with pytest.raises(ValidationError):
        search.Configuration(**search_config)


def test_api(tmp_path, search_config, search_output):

    from kiwi.lib.search import search_from_file

    train_dir = tmp_path / 'runs'
    output_dir = tmp_path / 'search'
    search_config['directory'] = output_dir

    config_file = tmp_path / 'config.yaml'

    search_config['base_config']['trainer']['gradient_accumulation_steps'] = 2
    search_config['base_config']['trainer']['main_metric'] = 'target_tags_MCC'
    search_config['base_config']['run']['use_mlflow'] = True
    search_config['base_config']['run']['output_dir'] = train_dir
    search_config['base_config']['system']['model']['outputs']['word_level'][
        'gaps'
    ] = True
    search_config['base_config']['system']['model']['outputs']['word_level'][
        'source'
    ] = True
    save_config_to_file(search.Configuration(**search_config), config_file)

    # Perfom a first search run
    search_from_file(config_file)
    # Check for search output and for training output
    assert set([file.name for file in tmp_path.glob('*')]) == set(
        ['config.yaml', 'search', 'runs']
    )
    assert [file.name for file in output_dir.glob('*')] == ['0']
    assert set(file.name for file in (output_dir / '0').glob('*')) == set(search_output)
    # Because `num_models_to_keep=1`, the second model should have been deleted
    assert len([file.name for file in train_dir.glob('checkpoints/*')]) == 1

    # Perfom a second search run, with different settings
    search_config['base_config']['run']['use_mlflow'] = False
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

    # Perfom a third search run, with yet different settings
    #   Create a folder that will clash with the next folder name...
    (output_dir / '3').mkdir(parents=True)
    #   ...and load the previous study.
    search_config['load_study'] = output_dir / '0' / 'study.pkl'
    save_config_to_file(search.Configuration(**search_config), config_file)
    search_from_file(config_file)
    folders = [file.name for file in output_dir.glob('*')]
    assert '0' in folders
    assert '1' in folders
    assert '3' in folders
    # Check the folder backup system
    assert sum(folder.startswith('3_backup_') for folder in folders) == 1


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
