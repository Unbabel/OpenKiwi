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

import pytest

from kiwi import __version__
from kiwi import constants
from kiwi.models.model import ModelConfig


@pytest.fixture
def state_dict():
    return {
        '__version__': __version__,
        'a': None,
        'b': 12,
        'c': 4,
        'source_vocab_size': 21,
        'target_vocab_size': 42,
    }


@pytest.fixture
def mock_vocab_one(state_dict):
    source_vocab_size = state_dict['source_vocab_size']
    target_vocab_size = state_dict['target_vocab_size']
    return {
        constants.SOURCE: range(source_vocab_size),
        constants.TARGET: range(target_vocab_size),
    }


@pytest.fixture
def mock_vocab_two():
    return {constants.SOURCE: range(1000), constants.TARGET: range(100)}


def test_set_vocab_sizes(mock_vocab_one):
    source_vocab_size = len(mock_vocab_one[constants.SOURCE])
    target_vocab_size = len(mock_vocab_one[constants.TARGET])
    config = ModelConfig(mock_vocab_one)
    assert config.source_vocab_size == source_vocab_size
    assert config.target_vocab_size == target_vocab_size


def test_from_dict(state_dict, mock_vocab_one):
    config = ModelConfig.from_dict(
        config_dict=state_dict, vocabs=mock_vocab_one
    )
    for k, v in state_dict.items():
        assert hasattr(config, k)
        assert getattr(config, k) == v


def test_update_config(state_dict, mock_vocab_one, mock_vocab_two):
    config_one = ModelConfig(vocabs=mock_vocab_two)
    config_two = ModelConfig.from_dict(
        config_dict=state_dict, vocabs=mock_vocab_one
    )
    config_one.update(config_two)
    for k, v in state_dict.items():
        assert hasattr(config_one, k)
        assert getattr(config_one, k) == v


def test_update_dict(state_dict, mock_vocab_one, mock_vocab_two):
    config_one = ModelConfig(vocabs=mock_vocab_two)
    config_two = ModelConfig.from_dict(
        config_dict=state_dict, vocabs=mock_vocab_one
    )
    config_one.update(config_two.__dict__)
    for k, v in state_dict.items():
        assert hasattr(config_one, k)
        assert getattr(config_one, k) == v


def test_state_dict(state_dict, mock_vocab_one):
    config = ModelConfig.from_dict(
        config_dict=state_dict, vocabs=mock_vocab_one
    )
    for k, v in config.state_dict().items():
        assert k in state_dict
        assert v == state_dict[k]
        del state_dict[k]
    assert not state_dict
