import pytest

from conftest import check_computation, check_jackknife
from kiwi import constants
from kiwi.models.nuqe import NuQE


@pytest.fixture
def wmt18_opts(nuqe_opts, data_opts_18):
    nuqe_opts.__dict__.update(data_opts_18.__dict__)
    return nuqe_opts


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
        NuQE,
        temp_output_dir,
        train_opts,
        target_opts,
        output_name=constants.TARGET_TAGS,
        expected_avg_probs=0.466939,
        atol=atol,
    )


def test_computation_gaps(temp_output_dir, train_opts, gap_opts, atol):
    check_computation(
        NuQE,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=constants.GAP_TAGS,
        expected_avg_probs=0.454558,
        atol=atol,
    )


def test_computation_source(temp_output_dir, train_opts, source_opts, atol):
    check_computation(
        NuQE,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=constants.SOURCE_TAGS,
        expected_avg_probs=0.474266,
        atol=atol,
    )


def test_jackknifing_target(temp_output_dir, train_opts, target_opts, atol):
    check_jackknife(
        NuQE,
        temp_output_dir,
        train_opts,
        target_opts,
        output_name=constants.TARGET_TAGS,
        expected_avg_probs=0.446874,
        atol=atol,
    )


def test_jackknifing_gaps(temp_output_dir, train_opts, gap_opts, atol):
    check_jackknife(
        NuQE,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=constants.GAP_TAGS,
        expected_avg_probs=0.463602,
        atol=atol,
    )


def test_jackknifing_source(temp_output_dir, train_opts, source_opts, atol):
    check_jackknife(
        NuQE,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=constants.SOURCE_TAGS,
        expected_avg_probs=0.464278,
        atol=atol,
    )


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
