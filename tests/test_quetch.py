from pathlib import Path

import numpy as np
import pytest

from conftest import check_computation, check_jackknife
from kiwi import constants as const
from kiwi.lib.utils import merge_namespaces, save_config_file
from kiwi.models.quetch import QUETCH


def test_computation(temp_output_dir, train_opts, quetch_opts, atol):
    check_computation(
        QUETCH,
        temp_output_dir,
        train_opts,
        quetch_opts,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.439731,
        atol=atol,
    )


def test_api(temp_output_dir, train_opts, quetch_opts, atol):
    from kiwi import train, load_model

    train_opts.model = 'quetch'
    train_opts.checkpoint_keep_only_best = 1
    all_opts = merge_namespaces(train_opts, quetch_opts)

    config_file = str(Path(temp_output_dir, 'config.yaml'))
    save_config_file(all_opts, config_file)

    assert train_opts.checkpoint_keep_only_best > 0

    train_run_info = train(config_file)

    predicter = load_model(train_run_info.model_path)

    examples = {
        const.SOURCE: open(quetch_opts.test_source).readlines(),
        const.TARGET: open(quetch_opts.test_target).readlines(),
        const.ALIGNMENTS: open(quetch_opts.test_alignments).readlines(),
    }

    predictions = predicter.predict(examples, batch_size=train_opts.batch_size)

    predictions = predictions[const.TARGET_TAGS]
    avg_of_avgs = np.mean(list(map(np.mean, predictions)))
    max_prob = max(map(max, predictions))
    min_prob = min(map(min, predictions))
    np.testing.assert_allclose(avg_of_avgs, 0.439731, atol=atol)
    assert 0 <= min_prob <= avg_of_avgs <= max_prob <= 1


def test_jackknifing(temp_output_dir, train_opts, quetch_opts, atol):
    check_jackknife(
        QUETCH,
        temp_output_dir,
        train_opts,
        quetch_opts,
        output_name=const.TARGET_TAGS,
        expected_avg_probs=0.486359,
        atol=atol,
    )


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
