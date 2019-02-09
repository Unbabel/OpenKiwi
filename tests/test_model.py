import collections
from types import SimpleNamespace

import numpy as np
import torch

from kiwi import constants as const
from kiwi.data.vocabulary import Vocabulary
from kiwi.models.model import Model


def test_get_mask():
    target_lengths = torch.LongTensor([1, 2, 3, 4])
    source_lengths = torch.LongTensor([4, 3, 2, 1])
    target_mask = [
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
    ]
    source_mask = [
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ]

    source_mask = torch.ByteTensor(source_mask)
    target_mask = torch.ByteTensor(target_mask)
    gap_mask = target_mask[:, 1:]
    target_tags_mask = target_mask[:, 1:-1]
    source_tags_mask = source_mask[:, 1:-1]
    source = torch.LongTensor(np.random.randint(4, 100, size=(4, 6)))
    target = torch.LongTensor(np.random.randint(4, 100, size=(4, 6)))
    source_tags = torch.LongTensor(np.random.randint(0, 2, size=(4, 4)))
    target_tags = torch.LongTensor(np.random.randint(0, 2, size=(4, 4)))
    gap_tags = torch.LongTensor(np.random.randint(0, 2, size=(4, 5)))

    source = source.masked_fill(1 - source_mask, const.PAD_ID)
    target = target.masked_fill(1 - target_mask, const.PAD_ID)
    target_tags = target_tags.masked_fill(
        1 - target_tags_mask, const.PAD_TAGS_ID
    )
    source_tags = source_tags.masked_fill(
        1 - source_tags_mask, const.PAD_TAGS_ID
    )
    gap_tags = gap_tags.masked_fill(1 - gap_mask, const.PAD_TAGS_ID)

    source[:, 0] = const.START_ID
    stop_mask = torch.arange(6).unsqueeze(0).expand_as(source) == (
        (source_lengths + 1).unsqueeze(1)
    )

    source = source.masked_fill(stop_mask, const.STOP_ID)
    target[:, 0] = const.START_ID
    stop_mask = torch.arange(6).unsqueeze(0).expand_as(target) == (
        (target_lengths + 1).unsqueeze(1)
    )
    target = target.masked_fill(stop_mask, const.STOP_ID)

    batch = SimpleNamespace(
        **{
            const.TARGET: target,
            const.SOURCE: source,
            const.TARGET_TAGS: target_tags,
            const.SOURCE_TAGS: source_tags,
            const.GAP_TAGS: gap_tags,
        }
    )

    vocab = Vocabulary(collections.Counter())
    vocab.stoi = {
        const.UNK: const.UNK_ID,
        const.PAD: const.PAD_ID,
        const.START: const.START_ID,
        const.STOP: const.STOP_ID,
    }
    tags_vocab = Vocabulary(collections.Counter())
    tags_vocab.stoi = {const.PAD: const.PAD_TAGS_ID}

    model = Model(
        vocabs={
            const.TARGET: vocab,
            const.SOURCE: vocab,
            const.TARGET_TAGS: tags_vocab,
            const.SOURCE_TAGS: tags_vocab,
            const.GAP_TAGS: tags_vocab,
        }
    )
    _source_mask = model.get_mask(batch, const.SOURCE)
    _target_mask = model.get_mask(batch, const.TARGET)
    _target_tags_mask = model.get_mask(batch, const.TARGET_TAGS)
    _source_tags_mask = model.get_mask(batch, const.SOURCE_TAGS)
    _gap_mask = model.get_mask(batch, const.GAP_TAGS)
    assert (_source_mask == source_mask).all()
    assert (_target_mask == target_mask).all()
    assert (_target_tags_mask == target_tags_mask).all()
    assert (_source_tags_mask == source_tags_mask).all()
    assert (_gap_mask == gap_mask).all()
