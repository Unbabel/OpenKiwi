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

from pathlib import Path

import numpy as np
import pytest

from conftest import check_computation, check_jackknife
from kiwi import constants as const
from kiwi.lib.utils import merge_namespaces, save_config_file
from kiwi.models.nuqe import NuQE


def test_computation(temp_output_dir, train_opts, nuqe_opts, atol):
    check_computation(
        NuQE,
        temp_output_dir,
        train_opts,
        nuqe_opts,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.572441,
        atol=atol,
    )


def test_api(temp_output_dir, train_opts, nuqe_opts, atol):
    from kiwi import train, load_model

    train_opts.model = 'nuqe'
    train_opts.checkpoint_keep_only_best = 1
    all_opts = merge_namespaces(train_opts, nuqe_opts)

    config_file = Path(temp_output_dir, 'config.yaml')
    save_config_file(all_opts, config_file)

    train_run_info = train(config_file)

    predicter = load_model(train_run_info.model_path)

    examples = {
        const.SOURCE: open(nuqe_opts.test_source).readlines(),
        const.TARGET: open(nuqe_opts.test_target).readlines(),
        const.ALIGNMENTS: open(nuqe_opts.test_alignments).readlines(),
    }

    predictions = predicter.predict(examples, batch_size=train_opts.batch_size)

    predictions = predictions[const.TARGET_TAGS]
    avg_of_avgs = np.mean(list(map(np.mean, predictions)))
    max_prob = max(map(max, predictions))
    min_prob = min(map(min, predictions))
    np.testing.assert_allclose(avg_of_avgs, 0.572441, atol=atol)
    assert 0 <= min_prob <= avg_of_avgs <= max_prob <= 1


def test_jackknifing(temp_output_dir, train_opts, nuqe_opts, atol):
    check_jackknife(
        NuQE,
        temp_output_dir,
        train_opts,
        nuqe_opts,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.583079,
        atol=atol,
    )


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
