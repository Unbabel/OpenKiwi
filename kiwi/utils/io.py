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
import logging
from pathlib import Path

import torch
from pydantic import BaseModel, Extra

logger = logging.getLogger(__name__)


class BaseConfig(BaseModel):
    """Base class for all pydantic configs. Used to configure base behaviour of configs.
    """

    class Config:
        # Throws an error whenever an extra key is provided, effectively making parsing,
        # strict
        extra = Extra.forbid


def default_map_location(storage, loc):
    return storage


def load_torch_file(file_path, map_location=None):
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f'Torch file not found: {file_path}')

    try:
        if map_location is None:
            map_location = default_map_location
        file_dict = torch.load(file_path, map_location=map_location)
    except ModuleNotFoundError as e:
        # Caused, e.g., by moving the Vocabulary or DefaultFrozenDict classes
        logger.info(
            'Trying to load a slightly outdated file and encountered an issue when '
            'unpickling; trying to work around it.'
        )
        if e.name == 'kiwi.data.utils':
            import sys
            from kiwi.utils import data_structures

            sys.modules['kiwi.data.utils'] = data_structures
            file_dict = torch.load(file_path, map_location=map_location)
            del sys.modules['kiwi.data.utils']
        elif e.name == 'torchtext':
            import sys
            from kiwi.data import vocabulary

            vocabulary.Vocab = vocabulary.Vocabulary
            sys.modules['torchtext'] = ''
            sys.modules['torchtext.vocab'] = vocabulary
            file_dict = torch.load(file_path, map_location=map_location)
            del sys.modules['torchtext.vocab']
            del sys.modules['torchtext']
        else:
            raise e

    if map_location is None:
        map_location = default_map_location
    file_dict = torch.load(file_path, map_location=map_location)

    return file_dict


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
            key = f'{prefix}.{key}'
        output_path = Path(directory, key)
        logger.info(f'Saving {key} predictions to {output_path}')
        save_file(output_path, preds, token_sep=' ', example_sep='\n')


def read_file(path):
    """Read a file into a list of lists of words."""
    with Path(path).open('r', encoding='utf8') as f:
        return [[token for token in line.strip().split()] for line in f]


def target_gaps_to_target(batch):
    """Extract target tags from wmt18 format file."""
    return batch[1::2]


def target_gaps_to_gaps(batch):
    """Extract gap tags from wmt18 format file."""
    return batch[::2]


def generate_slug(text, delimiter="-"):
    """Convert text to a normalized "slug" without whitespace.

    Borrowed from the nice https://humanfriendly.readthedocs.io, by Peter Odding.

    Arguments:
        text: the original text, for example ``Some Random Text!``.
        delimiter: the delimiter to use for separating words
                   (defaults to the ``-`` character).

    Return:
        the slug text, for example ``some-random-text``.

    Raise:
        :exc:`~exceptions.ValueError` when the provided text is nonempty but results
            in an empty slug.
    """
    import re

    slug = text.lower()
    escaped = delimiter.replace("\\", "\\\\")
    slug = re.sub("[^a-z0-9]+", escaped, slug)
    slug = slug.strip(delimiter)
    if text and not slug:
        msg = "The provided text %r results in an empty slug!"
        raise ValueError(format(msg, text))
    return slug
