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
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generic, Optional, TypeVar

from pydantic import PositiveInt, validator
from pydantic.generics import GenericModel
from typing_extensions import Literal

import kiwi.constants as const
from kiwi.data.batch import MultiFieldBatch
from kiwi.data.datasets.parallel_dataset import ParallelDataset
from kiwi.data.encoders.base import DataEncoders
from kiwi.data.encoders.field_encoders import TextEncoder
from kiwi.utils.io import BaseConfig, load_torch_file

logger = logging.getLogger(__name__)

T = TypeVar('T')


class InputFields(GenericModel, Generic[T]):
    source: T
    target: T


class EmbeddingsConfig(BaseConfig):
    """Paths to word embeddings file for each input field."""

    source: Optional[Path]
    target: Optional[Path]

    format: Literal['polyglot', 'word2vec', 'fasttext', 'glove', 'text'] = 'polyglot'
    """Word embeddings format. See README for specific formatting instructions."""


class VocabularyConfig(BaseConfig):
    min_frequency: InputFields[PositiveInt] = 1
    """Only add to vocabulary words that occur more than this number of times in the
    training dataset (doesn't apply to loaded or pretrained vocabularies)."""

    max_size: InputFields[Optional[PositiveInt]] = None
    """Only create vocabulary with up to this many words (doesn't apply to loaded or
    pretrained vocabularies)."""

    keep_rare_words_with_embeddings = False
    """Keep words that occur less then min-frequency but are
    in embeddings vocabulary."""

    add_embeddings_vocab = False
    """Add words from embeddings vocabulary to source/target vocabulary."""

    # TODO: add this feature
    # source_max_length: PositiveInt = float("inf")
    # 'Maximum source sequence length'
    # source_min_length: int = 1
    # 'Truncate source sequence length.'
    # target_max_length: PositiveInt = float("inf")
    # 'Maximum target sequence length to keep.'
    # target_min_length: int = 1
    # 'Truncate target sequence length.'

    @validator('min_frequency', 'max_size', pre=True, always=True)
    def check_nested_options(cls, v):
        if isinstance(v, int) or v is None:
            return {'source': v, 'target': v}
        return v


class ParallelDataEncoder(DataEncoders):
    class Config(BaseConfig):
        share_input_fields_encoders: bool = False
        """Share encoding/vocabs between source and target fields."""

        vocab: VocabularyConfig = VocabularyConfig()
        embeddings: Optional[EmbeddingsConfig] = None

        @validator('embeddings', pre=True, always=False)
        def warn_missing_feature(cls, v):
            if v is not None:
                raise ValueError('Embeddings are not yet supported in the new version.')
            return None

    def __init__(self, config: Config, field_encoders: Dict[str, TextEncoder] = None):
        super().__init__()
        self._vocabularies_initialized = False
        self.config = config

        self.field_encoders = {}

        if field_encoders is None:
            field_encoders = {}

        # Inputs
        if const.SOURCE in field_encoders and const.TARGET in field_encoders:
            self.field_encoders[const.SOURCE] = field_encoders[const.SOURCE]
            self.field_encoders[const.TARGET] = field_encoders[const.TARGET]
        elif const.SOURCE not in field_encoders and const.TARGET not in field_encoders:
            self.field_encoders[const.TARGET] = TextEncoder()
            if config.share_input_fields_encoders:
                self.field_encoders[const.SOURCE] = self.field_encoders[const.TARGET]
            else:
                self.field_encoders[const.SOURCE] = TextEncoder()
        elif const.TARGET in field_encoders:
            self.field_encoders[const.TARGET] = field_encoders[const.TARGET]
            if config.share_input_fields_encoders:
                self.field_encoders[const.SOURCE] = self.field_encoders[const.TARGET]
            else:
                raise ValueError(
                    f'QE requires encoders for {const.SOURCE} and {const.TARGET} '
                    f'fields, but only {const.TARGET} was provided and share vocab '
                    f'is false.'
                )
        elif const.SOURCE in field_encoders:
            self.field_encoders[const.SOURCE] = field_encoders[const.SOURCE]
            if config.share_input_fields_encoders:
                self.field_encoders[const.TARGET] = self.field_encoders[const.SOURCE]
            else:
                raise ValueError(
                    f'QE requires encoders for {const.SOURCE} and {const.TARGET} '
                    f'fields, but only {const.SOURCE} was provided and share vocab '
                    f'is false.'
                )

    def fit_vocabularies(self, dataset: ParallelDataset):
        # Inputs
        if self.config.share_input_fields_encoders:
            target_samples = dataset[const.TARGET] + dataset[const.SOURCE]
            source_samples = None
        else:
            target_samples = dataset[const.TARGET]
            source_samples = dataset[const.SOURCE]

        if self.field_encoders[const.TARGET].vocab is not None:
            logger.warning(
                f'Vocabulary for {const.TARGET} already exists; '
                f'not going to fit it to data'
            )
        else:
            self.field_encoders[const.TARGET].fit_vocab(
                target_samples,
                vocab_size=self.config.vocab.max_size.target,
                vocab_min_freq=self.config.vocab.min_frequency.target,
            )
        if source_samples:
            if self.field_encoders[const.SOURCE].vocab is not None:
                logger.warning(
                    f'Vocabulary for {const.SOURCE} already exists; '
                    f'not going to fit it to data'
                )
            else:
                self.field_encoders[const.SOURCE].fit_vocab(
                    source_samples,
                    vocab_size=self.config.vocab.max_size.source,
                    vocab_min_freq=self.config.vocab.min_frequency.source,
                )

        self._vocabularies_initialized = True

    def load_vocabularies(self, load_vocabs_from: Path = None, overwrite: bool = False):
        """Load serialized Vocabularies from disk into fields."""
        logger.info(f'Loading vocabularies from: {load_vocabs_from}')
        vocabs_dict = load_torch_file(load_vocabs_from)
        if const.VOCAB not in vocabs_dict:
            raise KeyError(f'File {load_vocabs_from} has no {const.VOCAB}')

        return self.vocabularies_from_dict(vocabs_dict[const.VOCAB], overwrite)

    def vocabularies_from_dict(self, vocabs_dict: Dict, overwrite: bool = False):
        # Inputs
        input_fields = [const.TARGET, const.SOURCE]
        for input_field in input_fields:
            if self.field_encoders[input_field].vocab is not None and not overwrite:
                logger.warning(
                    f'Vocabulary for {input_field} already exists; '
                    f'not going to fit it to data'
                )
            else:
                self.field_encoders[input_field].vocab = vocabs_dict[input_field]

        extraneous_vocabs = [
            vocab for vocab in vocabs_dict if vocab not in input_fields
        ]
        if extraneous_vocabs:
            logger.warning(
                f'Unknown vocabularies and thus not loaded: {extraneous_vocabs}'
            )

        self._vocabularies_initialized = True
        return self.vocabularies

    @property
    def vocabularies(self):
        """Return the vocabularies for all encoders that have one.

        Return:
            A dict mapping encoder names to Vocabulary instances.
        """
        if not self._vocabularies_initialized:
            return None

        vocabs = {}
        for name, encoder in self.field_encoders.items():
            if encoder is not None and encoder.vocabulary:
                vocabs[name] = encoder.vocabulary
        return vocabs

    def collate_fn(self, samples, device=None):
        if not self._vocabularies_initialized:
            raise ValueError(
                f'Vocabularies have not been initialized; need to first call '
                f'``{self.__class__.__name__}'
                f'.fit_vocabularies({ParallelDataset.__name__})`` or'
                f'``{self.__class__.__name__}.load_vocabularies(Path)``'
            )

        columns_data = defaultdict(list)
        for sample in samples:
            for column, value in sample.items():
                # data = self.data_encoders[column].encode(value)
                columns_data[column].append(value)

        batch = {
            column: self.field_encoders[column].batch_encode(examples)
            for column, examples in columns_data.items()
        }

        batch = MultiFieldBatch(batch)

        return batch
