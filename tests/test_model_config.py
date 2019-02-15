import pytest

from kiwi import constants as const
from kiwi.models.model import ModelConfig


class MockVocab(object):

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def token_to_id(self, token):
        return 0


@pytest.fixture
def state_dict():
    return {
        'a': None,
        'b': 12,
        'c': 4,
        'vocab_sizes': {const.SOURCE: 21, const.TARGET: 42},
        'pad_idx': {const.SOURCE: 0, const.TARGET: 0},
        'start_idx': {const.SOURCE: 0, const.TARGET: 0},
        'stop_idx': {const.SOURCE: 0, const.TARGET: 0}
    }


@pytest.fixture
def mock_vocab_one(state_dict):
    print(state_dict)
    source_vocab_size = state_dict['vocab_sizes'][const.SOURCE]
    target_vocab_size = state_dict['vocab_sizes'][const.TARGET]
    return {
        const.SOURCE: MockVocab(source_vocab_size),
        const.TARGET: MockVocab(target_vocab_size),
    }


@pytest.fixture
def mock_vocab_two():
    return {const.SOURCE: MockVocab(1000), const.TARGET: MockVocab(100)}


def test_set_vocab_sizes(mock_vocab_one):
    source_vocab_size = len(mock_vocab_one[const.SOURCE])
    target_vocab_size = len(mock_vocab_one[const.TARGET])
    print(len(mock_vocab_one[const.SOURCE]))
    config = ModelConfig(mock_vocab_one)
    print(config.vocab_sizes)
    assert config.vocab_sizes[const.SOURCE] == source_vocab_size
    assert config.vocab_sizes[const.TARGET] == target_vocab_size


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
    for k, v in config_one.state_dict().items():
        assert k in state_dict
        if isinstance(v, dict):
            for _k, _v in v.items():
                assert _k in state_dict[k]
                assert _v == state_dict[k][_k]
        else:
            assert v == state_dict[k]
        del state_dict[k]
    assert not state_dict


def test_state_dict(state_dict, mock_vocab_one):
    config = ModelConfig.from_dict(
        config_dict=state_dict, vocabs=mock_vocab_one
    )
    for k, v in config.state_dict().items():
        assert k in state_dict
        if isinstance(v, dict):
            for _k, _v in v.items():
                assert _k in state_dict[k]
                assert _v == state_dict[k][_k]
        else:
            assert v == state_dict[k]
        del state_dict[k]
    assert not state_dict
