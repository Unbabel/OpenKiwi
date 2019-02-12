import pytest
import numpy as np

from kiwi.data.builders import build_training_datasets, build_test_dataset
from kiwi.data.fieldsets.quetch import build_fieldset
from kiwi.data.qe_dataset import QEDataset


@pytest.fixture
def data_opts_no_validation(data_opts_17):
    data_opts_17.valid_source = data_opts_17.valid_target = None
    data_opts_17.test_source = data_opts_17.test_target = None
    data_opts_17.split = 0.9
    return data_opts_17


def test_build_train_datasets_no_valid(data_opts_no_validation, atol):
    datasets = build_training_datasets(
        fieldset=build_fieldset(), **vars(data_opts_no_validation)
    )
    assert(len(datasets) == 2)
    for dataset in datasets:
        assert(type(dataset) == QEDataset)
    train_size, dev_size = len(datasets[0]), len(datasets[1])
    np.testing.assert_allclose(train_size / (train_size + dev_size),
                               data_opts_no_validation.split,
                               atol=atol)


def test_build_train_datasets_valid(data_opts_17):
    datasets = build_training_datasets(
        fieldset=build_fieldset(), **vars(data_opts_17)
    )
    assert(len(datasets) == 2)
    for dataset in datasets:
        assert(type(dataset) == QEDataset)


def test_build_test_dataset(data_opts_17):
    dataset = build_test_dataset(
        fieldset=build_fieldset(), **vars(data_opts_17)
    )
    assert(type(dataset) == QEDataset)


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
