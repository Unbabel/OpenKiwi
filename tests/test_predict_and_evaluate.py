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

from kiwi import load_system
from kiwi.data.datasets.wmt_qe_dataset import WMTQEDataset
from kiwi.lib import predict


def test_predicting_and_evaluating(tmp_path, data_config, model_dir, atol):
    load_model = model_dir / 'nuqe.ckpt'

    predicting_config = predict.Configuration(
        run=dict(seed=42, output_dir=tmp_path, predict_on_data_partition='valid'),
        data=WMTQEDataset.Config(**data_config),
        system=dict(load=load_model),
        use_gpu=False,
    )

    predictions, metrics = predict.run(predicting_config, tmp_path)
    assert set(predictions.keys()) == {
        'target_tags',
        'target_tags_labels',
        'gap_tags',
        'gap_tags_labels',
        'sentence_scores',
        'sentence_scores_extras',
        'targetgaps_tags',
        'targetgaps_tags_labels',
    }
    assert (
        abs(metrics.word_scores['targetgaps_tags']['F1_Mult'][0] - 0.1937725747540829)
        < 0.1
    )
    assert (
        abs(metrics.word_scores['target_tags']['F1_Mult'][0] - 0.053539393196874105)
        < 0.1
    )
    assert abs(metrics.sentence_scores['scoring'][0][1] - -0.13380020964645983) < 0.1


def test_runner(tmp_path, data_config, model_dir, atol):
    load_model = model_dir / 'nuqe.ckpt'
    runner = load_system(load_model)

    data_config = WMTQEDataset.Config(**data_config)
    dataset = WMTQEDataset.build(data_config, valid=True)

    predictions = runner.predict(
        source=dataset['source'],
        target=dataset['target'],
        alignments=dataset['alignments'],
    )

    target_lengths = [len(s.split()) for s in dataset['target']]
    predicted_lengths = [len(s) for s in predictions.target_tags_BAD_probabilities]
    predicted_labels_lengths = [len(s) for s in predictions.target_tags_labels]

    assert target_lengths == predicted_lengths == predicted_labels_lengths

    predicted_gap_lengths = [len(s) - 1 for s in predictions.gap_tags_BAD_probabilities]
    predicted_gap_labels_len = [len(s) - 1 for s in predictions.gap_tags_labels]

    assert target_lengths == predicted_gap_lengths == predicted_gap_labels_len

    assert len(dataset['target']) == len(predictions.sentences_hter)


def test_predict_with_empty_sentences(data_config, model_dir):
    load_model = model_dir / 'nuqe.ckpt'
    runner = load_system(load_model)

    data_config = WMTQEDataset.Config(**data_config)
    dataset = WMTQEDataset.build(data_config, valid=True)

    test_data = dict(
        source=dataset['source'],
        target=dataset['target'],
        alignments=dataset['alignments'],
    )

    blank_indices = [1, 3, -1]

    for field in test_data:
        for idx in reversed(blank_indices):
            test_data[field].insert(idx, '')

    assert len(test_data['source']) == len(dataset['source']) + len(blank_indices)
    assert len(test_data['target']) == len(dataset['target']) + len(blank_indices)

    predictions = runner.predict(**test_data)

    assert len(predictions.target_tags_labels) == len(test_data['target'])
    assert len(predictions.target_tags_BAD_probabilities) == len(test_data['target'])


def test_predict_with_all_empty_sentences(data_config, model_dir):
    load_model = model_dir / 'nuqe.ckpt'
    runner = load_system(load_model)

    # All empty
    test_data = dict(source=[''] * 10, target=[''] * 10, alignments=[''] * 10,)
    predictions = runner.predict(**test_data)
    assert all(prediction is None for prediction in vars(predictions).values())


def test_predict_with_one_side_all_empty_sentence(data_config, model_dir):
    load_model = model_dir / 'nuqe.ckpt'
    runner = load_system(load_model)

    # One side all empty
    test_data = dict(source=['AB'] * 10, target=[''] * 10, alignments=[''] * 10,)
    with pytest.raises(ValueError, match='Received empty'):
        runner.predict(**test_data)

    # Other side
    test_data = dict(source=[''] * 10, target=['AB'] * 10, alignments=[''] * 10,)
    with pytest.raises(ValueError, match='Received empty'):
        runner.predict(**test_data)
