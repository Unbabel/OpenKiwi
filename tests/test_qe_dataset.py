import pytest

from kiwi import constants as const
from kiwi.data.builders import build_training_datasets
from kiwi.data.fieldsets.quetch import build_fieldset
from kiwi.data.iterators import build_bucket_iterator


def check_qe_dataset(options):
    train_dataset, dev_dataset = build_training_datasets(
        fieldset=build_fieldset(), **vars(options)
    )

    train_iter = build_bucket_iterator(
        train_dataset, batch_size=8, is_train=True, device=None
    )

    dev_iter = build_bucket_iterator(
        dev_dataset, batch_size=8, is_train=False, device=None
    )

    pad_idx = train_dataset.fields[const.SOURCE].vocab.token_to_id(const.PAD)

    for batch_train, batch_dev in zip(train_iter, dev_iter):
        train_source = getattr(batch_train, const.SOURCE)
        if isinstance(train_source, tuple):
            train_source, lenghts = train_source
        train_source.t()
        train_prev_len = train_source.shape[1]
        # buckets should be sorted in decreasing length order
        # so we can use pack/padded sequences
        for train_sample in train_source:
            train_mask = (train_sample != pad_idx)
            train_cur_len = train_mask.int().sum().item()
            assert train_cur_len <= train_prev_len
            train_prev_len = train_cur_len

    source_field = train_dataset.fields[const.SOURCE]
    target_field = train_dataset.fields[const.TARGET]
    target_tags_field = train_dataset.fields[const.TARGET_TAGS]
    # check if each token is in the vocab
    for train_sample in train_dataset:
        for word in getattr(train_sample, const.SOURCE):
            assert word in source_field.vocab.stoi
        for word in getattr(train_sample, const.TARGET):
            assert word in target_field.vocab.stoi
        for tag in getattr(train_sample, const.TARGET_TAGS):
            assert tag in target_tags_field.vocab.stoi


def test_qe_dataset(data_opts_17):
    check_qe_dataset(data_opts_17)


def test_qe_dataset_wmt18(data_opts_18):
    check_qe_dataset(data_opts_18)


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
