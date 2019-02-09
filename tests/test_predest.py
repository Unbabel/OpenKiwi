import argparse

import pytest

from conftest import check_computation, check_jackknife
from kiwi import constants
from kiwi.models.predictor_estimator import Estimator


@pytest.fixture
def wmt18_opts(predest_opts, data_opts_18):
    predest_opts.__dict__.update(data_opts_18.__dict__)
    return predest_opts


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
        Estimator,
        temp_output_dir,
        train_opts,
        target_opts,
        output_name=constants.TARGET_TAGS,
        expected_avg_probs=0.438163,
        atol=atol,
    )

    # Testing resuming training
    resume_opts = argparse.Namespace(**vars(train_opts))
    resume_opts.save_model = False
    resume_opts.checkpoint_save = True
    resume_opts.resume = True
    resume_opts.epochs += 1
    check_computation(
        Estimator,
        temp_output_dir,
        resume_opts,
        target_opts,
        output_name=constants.TARGET_TAGS,
        expected_avg_probs=0.438163,
        atol=atol,
    )


def test_computation_gaps(temp_output_dir, train_opts, gap_opts, atol):
    gap_opts.predict_target = False
    check_computation(
        Estimator,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=constants.GAP_TAGS,
        expected_avg_probs=0.293252,
        atol=atol,
    )
    gap_opts.predict_target = True
    check_computation(
        Estimator,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=constants.GAP_TAGS,
        expected_avg_probs=0.293695,
        atol=atol,
    )


def test_computation_source(temp_output_dir, train_opts, source_opts, atol):
    check_computation(
        Estimator,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=constants.SOURCE_TAGS,
        expected_avg_probs=0.432427,
        atol=atol,
    )


def test_jackknifing_target(temp_output_dir, train_opts, target_opts, atol):
    check_jackknife(
        Estimator,
        temp_output_dir,
        train_opts,
        target_opts,
        output_name=constants.TARGET_TAGS,
        expected_avg_probs=0.48883,
        atol=atol,
    )


def test_jackknifing_gaps(temp_output_dir, train_opts, gap_opts, atol):
    check_jackknife(
        Estimator,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=constants.GAP_TAGS,
        expected_avg_probs=0.387553,
        atol=atol,
    )


def test_jackknifing_source(temp_output_dir, train_opts, source_opts, atol):
    check_jackknife(
        Estimator,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=constants.SOURCE_TAGS,
        expected_avg_probs=0.467527,
        atol=atol,
    )


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
