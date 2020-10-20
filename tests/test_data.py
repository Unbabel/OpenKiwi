#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2020 Unbabel <openkiwi@unbabel.com>
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
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from kiwi import constants as const
from kiwi.data.batch import BatchedSentence, MultiFieldBatch, tensors_to
from kiwi.data.datasets.parallel_dataset import ParallelDataset
from kiwi.data.datasets.wmt_qe_dataset import WMTQEDataset
from kiwi.data.encoders.parallel_data_encoder import ParallelDataEncoder
from kiwi.data.encoders.wmt_qe_data_encoder import WMTQEDataEncoder


@pytest.fixture
def batched_sentence():
    return BatchedSentence(
        tensor=torch.zeros(1),
        lengths=torch.zeros(1),
        bounds=torch.zeros(1),
        bounds_lengths=torch.zeros(1),
        strict_masks=torch.zeros(1),
        number_of_tokens=torch.zeros(1),
    )


@pytest.fixture
def batch_dict(batched_sentence):
    return {'batch': batched_sentence}


@pytest.fixture
def multifield_batch(batch_dict):
    return MultiFieldBatch(batch_dict)


def test_build_training_dataset(data_config):
    config = WMTQEDataset.Config(**data_config)
    data_sources = WMTQEDataset.build(config=config, train=True, valid=True)
    assert len(data_sources) == 2
    for dataset in data_sources:
        assert type(dataset) == WMTQEDataset
    assert len(data_sources[0]) == 100
    assert len(data_sources[1]) == 50

    # TODO: put assert here
    data_sources[1].sort_key()
    # Some error checks:
    data_config['valid'] = None
    with pytest.raises(ValueError):
        config = WMTQEDataset.Config(**data_config)
    with pytest.raises(NotImplementedError):
        data_sources = WMTQEDataset.build(
            config=config, train=True, valid=False, split=0.1
        )


def test_build_test_dataset(data_config):
    config = WMTQEDataset.Config(**data_config)
    test_datasource = WMTQEDataset.build(config=config, test=True)
    assert type(test_datasource) == WMTQEDataset
    assert len(test_datasource) == 50


def test_parallel_dataset(data_config):
    config = ParallelDataset.Config(**data_config)
    datasets = ParallelDataset.build(config=config, train=True, valid=True, test=True)
    data_encoders = ParallelDataEncoder(config=ParallelDataEncoder.Config())
    train_dataset = datasets[0]
    data_encoders.fit_vocabularies(train_dataset)

    # TODO: put assert here
    train_dataset.sort_key()
    # Some error checks:
    data_config['valid'] = None
    with pytest.raises(ValueError):
        ParallelDataset.Config(**data_config)
    with pytest.raises(NotImplementedError):
        ParallelDataset.build(config=config, train=True, valid=False, split=0.1)

    sampler = BatchSampler(
        SequentialSampler(train_dataset), batch_size=8, drop_last=False
    )
    train_iter = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=data_encoders.collate_fn,
        pin_memory=torch.cuda.is_initialized(),
    )

    for batch in train_iter:
        assert batch['target'].strict_masks.sum(1).allclose(batch['target'].lengths - 2)

    sample = train_dataset[0]
    encoded = data_encoders.field_encoders[const.SOURCE].encode(sample[const.SOURCE])
    assert len(encoded) == 4
    assert encoded[0].tolist() == [
        2,
        5,
        182,
        13,
        5,
        193,
        154,
        111,
        5,
        462,
        625,
        150,
        5,
        183,
        6,
        3,
    ]
    encoded = data_encoders.field_encoders[const.TARGET].encode(sample[const.TARGET])
    assert len(encoded) == 4
    assert encoded[0].tolist() == [
        2,
        10,
        158,
        31,
        194,
        236,
        576,
        10,
        634,
        388,
        172,
        14,
        343,
        6,
        3,
    ]

    # Now initialize the ParallelDataEncoder from a field encoder
    data_encoder = ParallelDataEncoder(
        config=ParallelDataEncoder.Config(), field_encoders=data_encoders.field_encoders
    )
    # Fit vocabularies
    for dataset in datasets:
        data_encoder.fit_vocabularies(dataset)
    # Load vocabularies
    # data_encoder.load_vocabularies()


def check_qe_dataset(data_config_dict):
    config = WMTQEDataset.Config(**data_config_dict)
    train_dataset = WMTQEDataset.build(config=config, train=True, valid=False)
    data_encoders = WMTQEDataEncoder(config=WMTQEDataEncoder.Config())
    data_encoders.fit_vocabularies(train_dataset)

    sampler = BatchSampler(
        SequentialSampler(train_dataset), batch_size=8, drop_last=False
    )
    train_iter = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=data_encoders.collate_fn,
        pin_memory=torch.cuda.is_initialized(),
    )

    for batch in train_iter:
        assert batch['target'].strict_masks.sum(1).allclose(batch['target'].lengths - 2)

    sample = train_dataset[0]
    encoded = data_encoders.field_encoders[const.SOURCE].encode(sample[const.SOURCE])
    assert len(encoded) == 4
    assert encoded[0].tolist() == [
        2,
        5,
        182,
        13,
        5,
        193,
        154,
        111,
        5,
        462,
        625,
        150,
        5,
        183,
        6,
        3,
    ]
    encoded = data_encoders.field_encoders[const.TARGET].encode(sample[const.TARGET])
    assert len(encoded) == 4
    assert encoded[0].tolist() == [
        2,
        10,
        158,
        31,
        194,
        236,
        576,
        10,
        634,
        388,
        172,
        14,
        343,
        6,
        3,
    ]
    encoded = data_encoders.field_encoders[const.TARGET_TAGS].encode(
        sample[const.TARGET_TAGS]
    )
    assert len(encoded) == 4
    assert encoded[0].tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]


def test_qe_dataset_wmt18(data_config):
    check_qe_dataset(data_config)


def test_batched_sentence(batched_sentence):
    assert batched_sentence == batched_sentence.to('cpu')
    # There's no CUDA to pin to:
    with pytest.raises(RuntimeError):
        batched_sentence.pin_memory()


def test_multifield_batch(multifield_batch):
    assert multifield_batch == multifield_batch.to('cpu')
    # There's no CUDA to pin to:
    with pytest.raises(RuntimeError):
        multifield_batch.pin_memory()


def test_tensors_to(batched_sentence, multifield_batch, batch_dict):
    assert batched_sentence == tensors_to(batched_sentence, 'cpu')
    assert multifield_batch == tensors_to(multifield_batch, 'cpu')
    assert batch_dict == tensors_to(batch_dict, 'cpu')


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
