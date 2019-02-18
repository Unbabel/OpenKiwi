import argparse

import pytest

from conftest import check_computation, check_jackknife
from kiwi import constants as const
from kiwi.models.predictor_estimator import Estimator


@pytest.fixture
def predest_opts_no_multitask(predest_opts, data_opts_18):
    predest_opts.__dict__.update(data_opts_18.__dict__)
    return predest_opts


@pytest.fixture
def predest_opts_multitask(predest_opts_no_multitask, dir_18):
    multitask_opts = predest_opts_no_multitask

    multitask_opts.train_pe = str(dir_18.joinpath('train.pe'))
    multitask_opts.valid_pe = str(dir_18.joinpath('dev.pe'))
    multitask_opts.test_pe = str(dir_18.joinpath('dev.pe'))
    multitask_opts.train_sentence_scores = str(dir_18.joinpath('train.hter'))
    multitask_opts.valid_sentence_scores = str(dir_18.joinpath('dev.hter'))
    multitask_opts.test_sentence_scores = str(dir_18.joinpath('dev.hter'))

    multitask_opts.token_level = True
    multitask_opts.binary_level = True
    multitask_opts.sentence_level = True

    return multitask_opts


@pytest.fixture
def target_opts(predest_opts_multitask):
    options = predest_opts_multitask
    options.predict_target = True
    options.predict_gaps = False
    options.predict_source = False
    return options


@pytest.fixture
def gap_opts(predest_opts_multitask):
    options = predest_opts_multitask
    options.predict_target = False
    options.predict_gaps = True
    options.predict_source = False
    return options


@pytest.fixture
def source_opts(predest_opts_multitask):
    options = predest_opts_multitask
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
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.452956,
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
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.452956,
        atol=atol,
    )


def test_computation_gaps(temp_output_dir, train_opts, gap_opts, atol):
    gap_opts.predict_target = False
    check_computation(
        Estimator,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=const.GAP_TAGS,
        expected_avg_probs=0.289914,
        atol=atol,
    )
    gap_opts.predict_target = True
    check_computation(
        Estimator,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=const.GAP_TAGS,
        expected_avg_probs=0.331126,
        atol=atol,
    )


def test_computation_source(temp_output_dir, train_opts, source_opts, atol):
    check_computation(
        Estimator,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=const.SOURCE_TAGS,
        expected_avg_probs=0.449527,
        atol=atol,
    )


def test_jackknifing_target(temp_output_dir, train_opts, target_opts, atol):
    check_jackknife(
        Estimator,
        temp_output_dir,
        train_opts,
        target_opts,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.47876,
        atol=atol,
    )


def test_jackknifing_gaps(temp_output_dir, train_opts, gap_opts, atol):
    check_jackknife(
        Estimator,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=const.GAP_TAGS,
        expected_avg_probs=0.407023,
        atol=atol,
    )


def test_jackknifing_source(temp_output_dir, train_opts, source_opts, atol):
    check_jackknife(
        Estimator,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=const.SOURCE_TAGS,
        expected_avg_probs=0.484402,
        atol=atol,
    )


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
