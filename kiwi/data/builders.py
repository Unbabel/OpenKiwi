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

from functools import partial
from pathlib import Path

from kiwi.data.corpus import Corpus
from kiwi.data.fieldsets import extend_vocabs_fieldset
from kiwi.data.fieldsets.fieldset import Fieldset
from kiwi.data.qe_dataset import QEDataset
from kiwi.data.utils import (
    build_vocabulary,
    filter_len,
    load_vocabularies_to_datasets,
)


def build_dataset(fieldset, prefix='', filter_pred=None, **kwargs):
    fields, files = fieldset.fields_and_files(prefix, **kwargs)
    examples = Corpus.from_files(fields=fields, files=files)
    dataset = QEDataset(
        examples=examples, fields=fields, filter_pred=filter_pred
    )
    return dataset


def build_training_datasets(
    fieldset,
    split=0.0,
    valid_source=None,
    valid_target=None,
    load_vocab=None,
    **kwargs,
):
    """Build a training and validation QE datasets.

    Required Args:
        fieldset (Fieldset): specific set of fields to be used (depends on
                             the model to be used).
        train_source: Train Source
        train_target: Train Target (MT)

    Optional Args (depends on the model):
        train_pe: Train Post-edited
        train_target_tags: Train Target Tags
        train_source_tags: Train Source Tags
        train_sentence_scores: Train HTER scores

        valid_source: Valid Source
        valid_target: Valid Target (MT)
        valid_pe: Valid Post-edited
        valid_target_tags: Valid Target Tags
        valid_source_tags: Valid Source Tags
        valid_sentence_scores: Valid HTER scores

        split (float): If no validation sets are provided, randomly sample
                       1 - split of training examples as validation set.

        target_vocab_size: Maximum Size of target vocabulary
        source_vocab_size: Maximum Size of source vocabulary
        target_max_length: Maximum length for target field
        target_min_length: Minimum length for target field
        source_max_length: Maximum length for source field
        source_min_length: Minimum length for source field
        target_vocab_min_freq: Minimum word frequency target field
        source_vocab_min_freq: Minimum word frequency source field
        load_vocab: Path to existing vocab file

    Returns:
        A training and a validation Dataset.
    """
    # TODO: improve handling these length options (defaults are set multiple
    # times).
    filter_pred = partial(
        filter_len,
        source_min_length=kwargs.get('source_min_length', 1),
        source_max_length=kwargs.get('source_max_length', float('inf')),
        target_min_length=kwargs.get('target_min_length', 1),
        target_max_length=kwargs.get('target_max_length', float('inf')),
    )
    train_dataset = build_dataset(
        fieldset, prefix=Fieldset.TRAIN, filter_pred=filter_pred, **kwargs
    )
    if valid_source and valid_target:
        valid_dataset = build_dataset(
            fieldset,
            prefix=Fieldset.VALID,
            filter_pred=filter_pred,
            valid_source=valid_source,
            valid_target=valid_target,
            **kwargs,
        )
    elif split:
        if not 0.0 < split < 1.0:
            raise Exception(
                'Invalid data split value: {}; it must be in the '
                '(0, 1) interval.'.format(split)
            )
        train_dataset, valid_dataset = train_dataset.split(split)
    else:
        raise Exception('Validation data not provided.')

    if load_vocab:
        vocab_path = Path(load_vocab)
        load_vocabularies_to_datasets(vocab_path, train_dataset, valid_dataset)

    # Even if vocab is loaded, we need to build the vocabulary
    # in case fields are missing
    datasets_for_vocab = [train_dataset]
    if kwargs.get('extend_source_vocab') or kwargs.get('extend_target_vocab'):
        vocabs_fieldset = extend_vocabs_fieldset.build_fieldset(fieldset)
        extend_vocabs_ds = build_dataset(vocabs_fieldset, **kwargs)
        datasets_for_vocab.append(extend_vocabs_ds)

    fields_vocab_options = fieldset.fields_vocab_options(**kwargs)
    build_vocabulary(fields_vocab_options, *datasets_for_vocab)

    return train_dataset, valid_dataset


def build_test_dataset(fieldset, load_vocab=None, **kwargs):
    """Build a test QE dataset.

    Args:
      fieldset (Fieldset): specific set of fields to be used (depends on
                           the model to be used.)
      load_vocab: A path to a saved vocabulary.

    Returns:
        A Dataset object.
    """

    test_dataset = build_dataset(fieldset, prefix=Fieldset.TEST, **kwargs)

    fields_vocab_options = fieldset.fields_vocab_options(**kwargs)
    if load_vocab:
        vocab_path = Path(load_vocab)
        load_vocabularies_to_datasets(vocab_path, test_dataset)
    else:
        build_vocabulary(fields_vocab_options, test_dataset)

    return test_dataset
