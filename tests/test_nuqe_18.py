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

from conftest import check_computation, check_jackknife
from kiwi import constants as const
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
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.485723,
        atol=atol,
    )


def test_computation_gaps(temp_output_dir, train_opts, gap_opts, atol):
    check_computation(
        NuQE,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=const.GAP_TAGS,
        expected_avg_probs=0.454558,
        atol=atol,
    )


def test_computation_source(temp_output_dir, train_opts, source_opts, atol):
    check_computation(
        NuQE,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=const.SOURCE_TAGS,
        expected_avg_probs=0.474266,
        atol=atol,
    )


def test_jackknifing_target(temp_output_dir, train_opts, target_opts, atol):
    check_jackknife(
        NuQE,
        temp_output_dir,
        train_opts,
        target_opts,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.477126,
        atol=atol,
    )


def test_jackknifing_gaps(temp_output_dir, train_opts, gap_opts, atol):
    check_jackknife(
        NuQE,
        temp_output_dir,
        train_opts,
        gap_opts,
        output_name=const.GAP_TAGS,
        expected_avg_probs=0.526033,
        atol=atol,
    )


def test_jackknifing_source(temp_output_dir, train_opts, source_opts, atol):
    check_jackknife(
        NuQE,
        temp_output_dir,
        train_opts,
        source_opts,
        output_name=const.SOURCE_TAGS,
        expected_avg_probs=0.471008,
        atol=atol,
    )


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
