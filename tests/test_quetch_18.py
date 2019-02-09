import pytest

from conftest import check_computation, check_jackknife
from kiwi import constants
from kiwi.models.quetch import QUETCH


@pytest.fixture
def wmt18_opts(quetch_opts, data_opts_18):
    quetch_opts.__dict__.update(data_opts_18.__dict__)
    return quetch_opts


@pytest.fixture
def target_opts(wmt18_opts):
    options = wmt18_opts
    options.predict_target = True
    options.predict_gaps = False
    options.predict_source = False
    return options


@pytest.fixture
def gap_opts(wmt18_opts):
    options = wmt18_opts
    options.predict_target = False
    options.predict_gaps = True
    options.predict_source = False
    return options


@pytest.fixture
def source_opts(wmt18_opts):
    options = wmt18_opts
    options.predict_target = False
    options.predict_gaps = False
    options.predict_source = True
    return options


def test_computation_target(temp_output_dir, train_opts, target_opts, atol):
    check_computation(
        QUETCH,
        temp_output_dir,
        train_opts,
        target_opts,
        output_name=constants.TARGET_TAGS,
        expected_avg_probs=0.361237,
        atol=atol,
    )


def test_computation_gaps(temp_output_dir, train_opts, gap_opts, atol):
    check_computation(
        QUETCH,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=constants.GAP_TAGS,
        expected_avg_probs=0.251563,
        atol=atol,
    )


def test_computation_source(temp_output_dir, train_opts, source_opts, atol):
    check_computation(
        QUETCH,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=constants.SOURCE_TAGS,
        expected_avg_probs=0.355306,
        atol=atol,
    )


def test_jackknifing_target(temp_output_dir, train_opts, target_opts, atol):
    check_jackknife(
        QUETCH,
        temp_output_dir,
        train_opts,
        target_opts,
        output_name=constants.TARGET_TAGS,
        expected_avg_probs=0.372357,
        atol=atol,
    )


def test_jackknifing_gaps(temp_output_dir, train_opts, gap_opts, atol):
    check_jackknife(
        QUETCH,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=constants.GAP_TAGS,
        expected_avg_probs=0.279053,
        atol=atol,
    )


def test_jackknifing_source(temp_output_dir, train_opts, source_opts, atol):
    check_jackknife(
        QUETCH,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=constants.SOURCE_TAGS,
        expected_avg_probs=0.394607,
        atol=atol,
    )


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
