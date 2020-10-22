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
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch.utils.data
from more_itertools import all_equal
from pydantic import FilePath, confloat, validator

import kiwi.constants as const
from kiwi.utils.io import BaseConfig

logger = logging.getLogger(__name__)


class InputConfig(BaseConfig):
    source: FilePath
    """Path to a corpus file in the source language."""
    target: FilePath
    """Path to a corpus file in the target language."""
    alignments: Optional[FilePath] = None
    """Path to alignments between source and target."""
    post_edit: Optional[FilePath] = None
    """Path to file containing post-edited target."""
    source_pos: Optional[FilePath] = None
    """Path to input file with POS tags for source."""
    target_pos: Optional[FilePath] = None
    """Path to input file with POS tags for source."""


class OutputConfig(BaseConfig):
    target_tags: Optional[FilePath] = None
    """Path to label file for target."""
    source_tags: Optional[FilePath] = None
    """Path to label file for source."""
    sentence_scores: Optional[FilePath] = None
    """Path to file containing sentence level scores (HTER)."""


class TrainingConfig(BaseConfig):
    input: InputConfig
    output: OutputConfig


class TestConfig(BaseConfig):
    input: InputConfig


class WMTQEDataset(torch.utils.data.Dataset):
    class Config(BaseConfig):
        buffer_size: int = None
        """Number of consecutive instances to be temporarily stored in
        the buffer, which will be used later for batching/bucketing."""

        train: TrainingConfig = None
        valid: TrainingConfig = None
        test: TestConfig = None

        split: Optional[confloat(gt=0.0, lt=1.0)] = None
        """Split train dataset in case that no validation set is given."""

        @validator('split', pre=True, always=True)
        def ensure_there_is_validation_data(cls, v, values):
            if v is None and values.get('valid') is None:
                raise ValueError(
                    'In data configuration: must specify a `split` value in (0.0, 1.0) '
                    'or pass options for validation data using `valid.*`.'
                )
            return v

    @staticmethod
    def build(
        config: Config, directory=None, train=False, valid=False, test=False, split=0
    ):
        """Build training, validation, and test datasets.

        Arguments:
            config: configuration object with file paths and processing flags;
                    check out the docs for :class:`Config`.
            directory: if provided and paths in configuration are not absolute, use it
                       to anchor them.
            train: whether to build the training dataset.
            valid: whether to build the validation dataset.
            test: whether to build the testing dataset.
            split (float): If no validation set is provided, randomly sample
                           :math:`1-split` of training examples as validation set.
        """
        if split:
            raise NotImplementedError('Splitting functionality is not implemented.')

        datasets = []
        if train:
            columns_and_files = {
                # Inputs
                const.SOURCE: config.train.input.source,
                const.TARGET: config.train.input.target,
                const.ALIGNMENTS: config.train.input.alignments,
                const.SOURCE_POS: config.train.input.source_pos,
                const.TARGET_POS: config.train.input.target_pos,
                const.PE: config.train.input.post_edit,
                # Outputs
                const.SOURCE_TAGS: config.train.output.source_tags,
                const.TARGET_TAGS: config.train.output.target_tags,
                const.SENTENCE_SCORES: config.train.output.sentence_scores,
            }
            columns = {}
            for column, filename in columns_and_files.items():
                if filename:
                    filename = Path(filename)
                    if directory and not filename.is_absolute():
                        filename = Path(directory, filename)
                    columns[column] = read_file(filename, reader=None)
            train_dataset = WMTQEDataset(columns)
            datasets.append(train_dataset)

        if valid:
            columns_and_files = {
                # Inputs
                const.SOURCE: config.valid.input.source,
                const.TARGET: config.valid.input.target,
                const.ALIGNMENTS: config.valid.input.alignments,
                const.SOURCE_POS: config.valid.input.source_pos,
                const.TARGET_POS: config.valid.input.target_pos,
                const.PE: config.valid.input.post_edit,
                # Outputs
                const.SOURCE_TAGS: config.valid.output.source_tags,
                const.TARGET_TAGS: config.valid.output.target_tags,
                const.SENTENCE_SCORES: config.valid.output.sentence_scores,
            }
            columns = {}
            for column, filename in columns_and_files.items():
                if filename:
                    filename = Path(filename)
                    if directory and not filename.is_absolute():
                        filename = Path(directory, filename)
                    columns[column] = read_file(filename, reader=None)
            valid_dataset = WMTQEDataset(columns)
            datasets.append(valid_dataset)

        if test:
            columns_and_files = {
                # Inputs
                const.SOURCE: config.test.input.source,
                const.TARGET: config.test.input.target,
                const.ALIGNMENTS: config.test.input.alignments,
                const.SOURCE_POS: config.test.input.source_pos,
                const.TARGET_POS: config.test.input.target_pos,
            }
            columns = {}
            for column, filename in columns_and_files.items():
                if filename:
                    filename = Path(filename)
                    if directory and not filename.is_absolute():
                        filename = Path(directory, filename)
                    columns[column] = read_file(filename, reader=None)
            test_dataset = WMTQEDataset(columns)
            datasets.append(test_dataset)

        if len(datasets) == 1:
            return datasets[0]
        return tuple(datasets)

    def __init__(self, columns: Dict[Any, Union[Iterable, List]]):
        self._length = None
        num_samples = [len(col) for col in columns.values()]
        if not all_equal(num_samples):
            col_num_samples = {name: len(samples) for name, samples in columns.items()}
            raise ValueError(
                'Number of samples from different encoders do not match: '
                '{}'.format(col_num_samples)
            )
        self._length = num_samples[0] if num_samples else 0
        self.columns = columns

    def __getitem__(
        self, index_or_field: Union[int, str]
    ) -> Union[List[Any], Dict[str, Any]]:
        """Get a row with data from all fields or all rows for a given field"""
        if isinstance(index_or_field, str):
            column = [sample for sample in self.columns[index_or_field]]
            return column
        row = {name: value[index_or_field] for name, value in self.columns.items()}
        return row

    def __len__(self):
        return self._length

    def __contains__(self, item):
        return item in self.columns

    def sort_key(self, field='source'):
        def sort(dataset, _field, x):
            return len(dataset[x][_field])

        return partial(sort, self, field)


def read_file(path, reader):
    if reader:
        data = reader(path)
    else:
        with Path(path).open('r', encoding='utf8') as f:
            data = [line.strip() for line in f]
    return data
