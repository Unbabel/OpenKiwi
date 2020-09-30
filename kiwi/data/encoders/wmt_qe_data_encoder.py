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

from pydantic import PositiveInt
from pydantic.class_validators import validator
from pydantic.generics import GenericModel
from typing_extensions import Literal

import kiwi.constants as const
from kiwi.data import tokenizers
from kiwi.data.batch import MultiFieldBatch
from kiwi.data.datasets.wmt_qe_dataset import WMTQEDataset
from kiwi.data.encoders.base import DataEncoders
from kiwi.data.encoders.field_encoders import (
    AlignmentEncoder,
    BinaryScoreEncoder,
    ScoreEncoder,
    TagEncoder,
    TextEncoder,
)
from kiwi.utils.io import (
    BaseConfig,
    load_torch_file,
    target_gaps_to_gaps,
    target_gaps_to_target,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class InputFields(GenericModel, Generic[T]):
    source: T
    target: T


class EmbeddingsConfig(BaseConfig):
    """Paths to word embeddings file for each input field."""

    source: Optional[Path]
    target: Optional[Path]
    post_edit: Optional[Path] = None
    source_pos: Optional[Path] = None
    target_pos: Optional[Path] = None

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
    'Keep words that occur less then min-frequency but are in embeddings vocabulary'

    add_embeddings_vocab = False
    'Add words from embeddings vocabulary to source/target vocabulary'

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


class WMTQEDataEncoder(DataEncoders):
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
        # TLM multi-task fine-tuning output
        if const.PE in field_encoders:
            self.field_encoders[const.PE] = field_encoders[const.PE]
        else:
            self.field_encoders[const.PE] = self.field_encoders[const.TARGET]

        # Word level outputs
        logger.debug('Assuming WMT18 format for target tags outputs')
        self.field_encoders[const.TARGET_TAGS] = TagEncoder(
            tokenize=lambda x: target_gaps_to_target(tokenizers.tokenize(x))
        )
        self.field_encoders[const.GAP_TAGS] = TagEncoder(
            tokenize=lambda x: target_gaps_to_gaps(tokenizers.tokenize(x))
        )
        self.field_encoders[const.SOURCE_TAGS] = TagEncoder()

        # Sentence level outputs
        # if const.SENTENCE_SCORES in data_source:
        self.field_encoders[const.SENTENCE_SCORES] = ScoreEncoder()
        self.field_encoders[const.BINARY] = BinaryScoreEncoder()

        # Optional fields
        # For optional fields, check whether they exist in data_sources
        # if const.TARGET_POS in data_source:
        self.field_encoders[const.ALIGNMENTS] = AlignmentEncoder()

    def fit_vocabularies(self, dataset: WMTQEDataset):
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
                # embeddings_name=self.config.embeddings,
                # keep_rare_words_with_embeddings=self.config.vocab.keep_rare_words_with_embeddings,  # NoQA
                # add_embeddings_vocab=self.config.vocab.add_embeddings_vocab,
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
                    # embeddings_name=self.config.embeddings,
                    # keep_rare_words_with_embeddings=self.config.vocab.keep_rare_words_with_embeddings,  # NoQA
                    # add_embeddings_vocab=self.config.vocab.add_embeddings_vocab,
                )

        # Outputs
        for tag_output in [const.TARGET_TAGS, const.SOURCE_TAGS]:
            if tag_output in dataset:
                if self.field_encoders[tag_output].vocab is not None:
                    logger.warning(
                        f'Vocabulary for {tag_output} already exists; '
                        f'not going to fit it to data'
                    )
                else:
                    self.field_encoders[tag_output].fit_vocab(dataset[tag_output])
        self.field_encoders[const.GAP_TAGS].vocab = self.field_encoders[
            const.TARGET_TAGS
        ].vocab

        if const.TARGET_TAGS in dataset:
            # With data, we can check whether we really have data in WMT18 format
            target_sample = dataset[const.TARGET][0]
            _, target_bounds, *_ = self.field_encoders[const.TARGET].encode(
                target_sample
            )
            extras = 1 if self.field_encoders[const.TARGET].bos_token else 0
            extras += 1 if self.field_encoders[const.TARGET].eos_token else 0
            target_sample_length = len(target_bounds) - extras

            tag_sample = dataset[const.TARGET_TAGS][0]
            tags, *_ = self.field_encoders[const.TARGET_TAGS].encode(tag_sample)
            if len(tags) == target_sample_length // 2:
                # Pre WMT18 format (old format; gap tags in a separate file)
                logger.debug('Inferred old WMT17 format for tags outputs')
                self.field_encoders[const.TARGET_TAGS] = TagEncoder()
                # Fit again
                self.field_encoders[const.TARGET_TAGS].fit_vocab(
                    dataset[const.TARGET_TAGS]
                )
                self.field_encoders[const.GAP_TAGS] = self.field_encoders[
                    const.TARGET_TAGS
                ]

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
        input_fields = [const.TARGET, const.SOURCE, const.PE]
        output_fields = [const.TARGET_TAGS, const.SOURCE_TAGS, const.GAP_TAGS]

        extraneous_fields = [
            field for field in vocabs_dict if field not in input_fields + output_fields
        ]
        if extraneous_fields:
            logger.warning(
                f'Unknown vocabularies and thus not loaded: {extraneous_fields}'
            )

        fields_for_loading = [
            field for field in vocabs_dict if field in input_fields + output_fields
        ]
        for field in fields_for_loading:
            if self.field_encoders[field].vocab is not None and not overwrite:
                logger.warning(
                    f'Vocabulary for {field} already exists; not loading it again'
                )
            else:
                self.field_encoders[field].vocab = vocabs_dict[field]

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
                f'.fit_vocabularies({WMTQEDataset.__name__})`` or'
                f'``{self.__class__.__name__}.load_vocabularies(Path)``'
            )

        columns_data = defaultdict(list)
        for sample in samples:
            for column, value in sample.items():
                # data = self.data_encoders[column].encode(value)
                columns_data[column].append(value)
            # FIXME: hack to have gap_tags in the batch
            if const.TARGET_TAGS in sample and const.GAP_TAGS not in sample:
                columns_data[const.GAP_TAGS].append(sample[const.TARGET_TAGS])
            if const.SENTENCE_SCORES in sample and const.BINARY in self.field_encoders:
                columns_data[const.BINARY].append(sample[const.SENTENCE_SCORES])

        batch = {
            column: self.field_encoders[column].batch_encode(examples)
            for column, examples in columns_data.items()
        }

        batch = MultiFieldBatch(batch)

        return batch
