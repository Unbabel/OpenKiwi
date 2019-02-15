from pathlib import Path

import numpy as np
import pytest

from kiwi import constants as const
from kiwi.data.builders import build_training_datasets
from kiwi.data.fieldsets import extend_vocabs_fieldset
from kiwi.lib import train
from kiwi.models.predictor import Predictor


@pytest.fixture
def extend_vocab(predictor_opts, predest_dir):
    options = predictor_opts
    options.extend_source_vocab = str(predest_dir.joinpath('extend.src'))
    options.extend_target_vocab = str(predest_dir.joinpath('extend.tgt'))
    return options


def test_extend_vocabs(extend_vocab):
    options = extend_vocab
    OOV_SRC = 'oov_word_src'
    OOV_TGT = 'oov_word_tgt'

    fieldset = Predictor.fieldset(wmt18_format=options.wmt18_format)
    vocabs_fieldset = extend_vocabs_fieldset.build_fieldset(fieldset)

    dataset, _ = build_training_datasets(
        fieldset, extend_vocabs_fieldset=vocabs_fieldset, **vars(options)
    )
    assert OOV_SRC in dataset.fields[const.SOURCE].vocab.stoi
    assert OOV_TGT in dataset.fields[const.TARGET].vocab.stoi

    fieldset = Predictor.fieldset(wmt18_format=options.wmt18_format)
    options.extend_source_vocab = None
    options.extend_target_vocab = None
    dataset, _ = build_training_datasets(fieldset, **vars(options))
    assert OOV_SRC not in dataset.fields[const.SOURCE].vocab.stoi
    assert OOV_TGT not in dataset.fields[const.TARGET].vocab.stoi


def test_train_predictor(temp_output_dir, predictor_opts, train_opts, atol):
    train_opts.save_model = temp_output_dir
    train_opts.load_model = None
    train_opts.checkpoint_keep_only_best = 0

    trainer = train.run(Predictor, temp_output_dir, train_opts, predictor_opts)
    stats = trainer.stats_summary_history[-1]
    np.testing.assert_allclose(stats['target_PERP'], 410.781112, atol=atol)

    # Testing predictor with pickled data
    epoch_dir = 'epoch_{}'.format(train_opts.epochs)
    train_opts.load_model = str(Path(temp_output_dir, epoch_dir, 'model.torch'))

    trainer = train.run(Predictor, temp_output_dir, train_opts, predictor_opts)
    stats = trainer.stats_summary_history[-1]
    np.testing.assert_allclose(stats['target_PERP'], 376.099405, atol=atol)


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
