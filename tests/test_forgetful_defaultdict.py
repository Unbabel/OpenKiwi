import pytest

from kiwi.data.forgetful_defaultdict import ForgetfulDefaultdict


@pytest.fixture
def default():
    return object()


@pytest.fixture
def some_keys():
    return set([0, 1, 2, 'a', 'b', 'c', object(), object()])


@pytest.fixture
def some_dict(some_keys):
    return {key: hash(key) for key in some_keys}


def test_init_from_dict(default, some_dict):
    ffdd = ForgetfulDefaultdict(default, some_dict)
    assert isinstance(ffdd, dict)
    for key, val in ffdd.items():
        assert key in some_dict
        assert some_dict[key] == val
    assert ffdd.keys() == some_dict.keys()


def test_saves_keys_and_returns_values(default, some_keys):
    ffdd = ForgetfulDefaultdict(default)
    for key in some_keys:
        assert key not in ffdd
        ffdd[key] = hash(key)
        assert key in ffdd
        assert ffdd[key] == hash(key)


def test_lookup_non_contained_values_returns_default(default, some_keys):
    ffdd = ForgetfulDefaultdict(default)
    some_key = some_keys.pop()
    assert ffdd[some_key] == default
    for key in some_keys:
        ffdd[key] = hash(key)
    assert ffdd[some_key] == default


def test_does_not_cache(default, some_keys):
    ffdd = ForgetfulDefaultdict(default)
    some_key = some_keys.pop()
    assert some_key not in ffdd
    ffdd[some_key]
    assert some_key not in ffdd
