import pytest

from kiwi import constants
from kiwi.models.model import ModelConfig


@pytest.fixture
def state_dict():
    return {
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
