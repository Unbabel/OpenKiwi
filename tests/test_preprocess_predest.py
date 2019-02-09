from pathlib import Path

import pytest
import torch

from kiwi import constants
from kiwi.lib import preprocess
from kiwi.models.predictor import Predictor


def test_preprocess_predest(temp_output_dir, predest_opts, general_opts):
    preprocess.run(Predictor, temp_output_dir, predest_opts)
    datafile = Path(temp_output_dir, constants.DATAFILE)
    assert datafile.exists()
    datasets = torch.load(str(datafile))
    assert constants.TRAIN in datasets
    assert constants.EVAL in datasets


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
