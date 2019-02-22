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

import copy
import logging
from collections import defaultdict
from math import ceil
from pathlib import Path

import torch

from kiwi import constants as const
from kiwi.data.fieldsets.fieldset import Fieldset

logger = logging.getLogger(__name__)


def serialize_vocabs(vocabs, include_vectors=False):
    """Make vocab dictionary serializable.
    """
    serialized_vocabs = []

    for name, vocab in vocabs.items():
        vocab = copy.copy(vocab)
        vocab.stoi = dict(vocab.stoi)
        if not include_vectors:
            vocab.vectors = None
        serialized_vocabs.append((name, vocab))

    return serialized_vocabs


def deserialize_vocabs(vocabs):
    """Restore defaultdict lost in serialization.
    """
    vocabs = dict(vocabs)
    for name, vocab in vocabs.items():
        # Hack. Can't pickle defaultdict :(
        vocab.stoi = defaultdict(lambda: const.UNK_ID, vocab.stoi)
    return vocabs


def serialize_fields_to_vocabs(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    From OpenNMT
    """
    vocabs = fields_to_vocabs(fields)
    vocabs = serialize_vocabs(vocabs)
    return vocabs


def deserialize_fields_from_vocabs(fields, vocabs):
    """
    Load serialized vocabularies into their fields.
    """
    # TODO redundant deserialization
    vocabs = deserialize_vocabs(vocabs)
    return fields_from_vocabs(fields, vocabs)


def fields_from_vocabs(fields, vocabs):
    """
    Load Field objects from vocabs dict.
    From OpenNMT
    """
    vocabs = deserialize_vocabs(vocabs)
    for name, vocab in vocabs.items():
        if name not in fields:
            logger.debug(
                'No field "{}" for loading vocabulary; ignoring.'.format(name)
            )
        else:
            fields[name].vocab = vocab
    return fields


def fields_to_vocabs(fields):
    """
    Extract Vocab Dictionary from Fields Dictionary.
       Args:
          fields: A dict mapping field names to Field objects
       Returns:
          vocab: A dict mapping field names to Vocabularies
    """
    vocabs = {}
    for name, field in fields.items():
        if field is not None and 'vocab' in field.__dict__:
            vocabs[name] = field.vocab
    return vocabs


def save_vocabularies_from_fields(directory, fields, include_vectors=False):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    From OpenNMT
    """
    vocabs = serialize_fields_to_vocabs(fields)
    vocab_path = Path(directory, const.VOCAB_FILE)
    torch.save({const.VOCAB: vocabs}, str(vocab_path))
    return vocab_path


def load_vocabularies_to_fields(vocab_path, fields):
    """Load serialized Vocabularies from disk into fields."""
    if Path(vocab_path).exists():
        vocabs_dict = torch.load(
            str(vocab_path), map_location=lambda storage, loc: storage
        )
        vocabs = vocabs_dict[const.VOCAB]
        fields = deserialize_fields_from_vocabs(fields, vocabs)
        logger.info('Loaded vocabularies from {}'.format(vocab_path))
        return all(
            [vocab_loaded_if_needed(field) for _, field in fields.items()]
        )
    return False


def load_vocabularies_to_datasets(vocab_path, *datasets):
    fields = {}
    for dataset in datasets:
        fields.update(dataset.fields)
    return load_vocabularies_to_fields(vocab_path, fields)


def vocab_loaded_if_needed(field):
    return not field.use_vocab or (hasattr(field, const.VOCAB) and field.vocab)


def save_vocabularies_from_datasets(directory, *datasets):
    fields = {}
    for dataset in datasets:
        fields.update(dataset.fields)
    return save_vocabularies_from_fields(directory, fields)


def build_vocabulary(fields_vocab_options, *datasets):
    fields = {}
    for dataset in datasets:
        fields.update(dataset.fields)

    for name, field in fields.items():
        if not vocab_loaded_if_needed(field):
            kwargs_vocab = fields_vocab_options[name]
            if 'vectors_fn' in kwargs_vocab:
                vectors_fn = kwargs_vocab['vectors_fn']
                kwargs_vocab['vectors'] = vectors_fn()
                del kwargs_vocab['vectors_fn']
            field.build_vocab(*datasets, **kwargs_vocab)


def load_datasets(directory, *datasets_names):
    dataset_path = Path(directory, const.DATAFILE)
    dataset_dict = torch.load(
        str(dataset_path), map_location=lambda storage, loc: storage
    )

    datasets = [dataset_dict[name] for name in datasets_names]
    return datasets


def save_datasets(directory, **named_datasets):
    """Pickle datasets to standard file in directory

    Note that fields cannot be saved as part of a dataset, so they are
    ignored.

    Args:
        directory (str or Path): directory where to save the datasets pickle.
        named_datasets (dict): mapping of name and respective dataset.
    """
    # Fields cannot be pickled
    # Saving field to a temporary list
    dataset_fields_tmp = []
    for dataset in named_datasets.values():
        dataset_fields_tmp.append(dataset.fields)
        dataset.fields = []

    logging.info('Saving preprocessed datasets...')
    dataset_path = Path(directory, const.DATAFILE)
    torch.save(named_datasets, str(dataset_path))

    # Reconstructing dataset.field from the temporary list
    for dataset, fields in zip(named_datasets.values(), dataset_fields_tmp):
        dataset.fields = fields


def save_training_datasets(directory, train_dataset, valid_dataset):
    ds_dict = {const.TRAIN: train_dataset, const.EVAL: valid_dataset}
    save_datasets(directory, **ds_dict)


def load_training_datasets(directory, fieldset):
    # FIXME: test if this works. Ideally, fields would be already contained
    # inside the loaded datasets.
    train_ds, valid_ds = load_datasets(directory, const.TRAIN, const.EVAL)

    # Remove fields not actually loaded (checking if they're required).
    fields = fieldset.fields
    for field in dict(fields):  # Make a copy so del can be used
        if not hasattr(train_ds.examples[0], field):
            for set_name in [Fieldset.TRAIN, Fieldset.VALID]:
                if fieldset.is_required(field, set_name):
                    raise AttributeError(
                        'Loaded {} dataset does not have a '
                        '{} field.'.format(set_name, field)
                    )
            del fields[field]

    train_ds.fields = fields
    valid_ds.fields = fields

    load_vocabularies_to_fields(
        Path(directory, const.VOCAB_FILE), fieldset.fields
    )

    return train_ds, valid_ds


def cross_split_dataset(dataset, splits):
    examples_per_split = ceil(len(dataset) / splits)
    for split in range(splits):
        held_out_start = examples_per_split * split
        held_out_stop = examples_per_split * (split + 1)

        held_out_examples = dataset[held_out_start:held_out_stop]
        held_in_examples = dataset[:held_out_start] + dataset[held_out_stop:]

        train_split = dataset.__class__(held_in_examples, dataset.fields)
        eval_split = dataset.__class__(held_out_examples, dataset.fields)

        yield train_split, eval_split


def save_file(file_path, data, token_sep=' ', example_sep='\n'):
    if data and isinstance(data[0], list):
        data = [token_sep.join(map(str, sentence)) for sentence in data]
    else:
        data = map(str, data)
    example_str = example_sep.join(data) + '\n'
    Path(file_path).write_text(example_str)


def save_predicted_probabilities(directory, predictions, prefix=''):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, preds in predictions.items():
        if prefix:
            key = '{}.{}'.format(prefix, key)
        output_path = Path(directory, key)
        logger.info('Saving {} predictions to {}'.format(key, output_path))
        save_file(output_path, preds, token_sep=' ', example_sep='\n')


def read_file(path):
    """Reads a file into a list of lists of words.
    """
    with Path(path).open('r', encoding='utf8') as f:
        return [[token for token in line.strip().split()] for line in f]


def hter_to_binary(x):
    """Transform hter score into binary OK/BAD label.
    """
    return ceil(float(x))


def wmt18_to_target(batch, *args):
    """Extract target tags from wmt18 format file.
    """
    return batch[1::2]


def wmt18_to_gaps(batch, *args):
    """Extract gap tags from wmt18 format file.
    """
    return batch[::2]


def project(batch, *args):
    """Projection onto the first argument.

       Needed to create a postprocessing pipeline that implements the identity.
    """
    return batch


def filter_len(
    x,
    source_min_length=1,
    source_max_length=float('inf'),
    target_min_length=1,
    target_max_length=float('inf'),
):
    return (source_min_length <= len(x.source) <= source_max_length) and (
        target_min_length <= len(x.target) <= target_max_length
    )
