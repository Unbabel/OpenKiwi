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
from collections.__init__ import Counter, OrderedDict
from typing import Optional

import numpy as np
import torch
from torchnlp.encoders.text import stack_and_pad_tensors

from kiwi.constants import PAD, START, STOP, UNALIGNED, UNK
from kiwi.data import tokenizers
from kiwi.data.batch import BatchedSentence
from kiwi.data.tokenizers import align_tokenize
from kiwi.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class TextEncoder:
    """Encode a field, handling vocabulary, tokenization and embeddings.

    Heavily inspired in torchtext and torchnlp.
    """

    def __init__(
        self,
        tokenize=tokenizers.tokenize,
        detokenize=tokenizers.detokenize,
        subtokenize=None,
        pad_token=PAD,
        unk_token=UNK,
        bos_token=START,
        eos_token=STOP,
        unaligned_token=UNALIGNED,
        specials_first=True,
        # extra options from fields?
        include_lengths=True,
        include_bounds=True,
    ):
        self.tokenize = tokenize
        self.detokenize = detokenize
        self.subtokenize = subtokenize

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unaligned_token = unaligned_token
        self.specials_first = specials_first

        self.include_lengths = include_lengths
        self.include_bounds = include_bounds

        self.tokens = None
        self.vocab: Optional[Vocabulary] = None

    def fit_vocab(
        self,
        samples,
        vocab_size=None,
        vocab_min_freq=0,
        embeddings_name=None,
        keep_rare_words_with_embeddings=False,
        add_embeddings_vocab=False,
    ):
        tokens = Counter()
        for sample in samples:
            # TODO: subtokenize?
            tokens.update(self.tokenize(sample))

        # We use our own Vocabulary class
        specials = list(
            OrderedDict.fromkeys(
                tok for tok in [self.unaligned_token] if tok is not None
            )
        )
        # TODO: handle embeddings/vectors
        self.vocab = Vocabulary(
            tokens,
            max_size=vocab_size,
            min_freq=vocab_min_freq,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            specials=specials,
            specials_first=self.specials_first,
            # TODO: missing vectors, etc.
            vectors=None,
            rare_with_vectors=keep_rare_words_with_embeddings,
            add_vectors_vocab=add_embeddings_vocab,
        )

    @property
    def vocabulary(self):
        return self.vocab

    @property
    def padding_index(self):
        return self.vocab.token_to_id(self.pad_token)

    def encode(self, example):
        tokens = self.tokenize(example)
        number_of_tokens = len(tokens)
        bounds = None
        if self.subtokenize is not None:
            token_pieces = []
            bounds = []
            offset = 0
            for token in tokens:
                pieces = self.subtokenize(token)
                token_pieces += pieces
                bounds.append(offset)
                offset += len(pieces)
            tokens = token_pieces
        if not bounds:
            bounds = list(range(len(tokens)))
        bounds = torch.tensor(bounds, dtype=torch.long)

        strict_mask = torch.ones(len(tokens), dtype=torch.bool)

        if self.bos_token is not None:
            tokens.insert(0, self.bos_token)
            bounds = torch.cat((torch.zeros(1, dtype=torch.long), bounds + 1))
            strict_mask = torch.cat([torch.tensor([False]), strict_mask])
        if self.eos_token is not None:
            tokens.append(self.eos_token)
            bounds = torch.cat(
                (bounds, torch.tensor([len(tokens) - 1], dtype=torch.long))
            )
            strict_mask = torch.cat([strict_mask, torch.tensor([False])])

        vector = [self.vocab.token_to_id(token) for token in tokens]

        return torch.tensor(vector), bounds, strict_mask, number_of_tokens

    def batch_encode(self, iterator):
        ids, bounds, strict_masks, number_of_tokens = list(
            zip(*[self.encode(example) for example in iterator])
        )
        batch = stack_and_pad_tensors(ids, padding_index=self.padding_index, dim=0)
        bounds_batch = stack_and_pad_tensors(bounds, padding_index=-1, dim=0)
        masks_batch = stack_and_pad_tensors(strict_masks, padding_index=False, dim=0)
        number_of_tokens_batch = torch.tensor(number_of_tokens, dtype=torch.int)

        return BatchedSentence(
            tensor=batch.tensor,
            lengths=batch.lengths,
            bounds=bounds_batch.tensor,
            bounds_lengths=bounds_batch.lengths,
            strict_masks=masks_batch.tensor,
            number_of_tokens=number_of_tokens_batch,
        )


class TagEncoder(TextEncoder):
    def __init__(
        self,
        tokenize=tokenizers.tokenize,
        detokenize=tokenizers.detokenize,
        pad_token=PAD,
        include_lengths=True,
    ):
        super().__init__(
            tokenize=tokenize,
            detokenize=detokenize,
            pad_token=pad_token,
            unk_token=None,
            bos_token=None,
            eos_token=None,
            unaligned_token=None,
            specials_first=False,
            include_lengths=include_lengths,
            include_bounds=False,
        )


class InputEncoder:
    def __init__(self):
        self.vocabulary = None


class ScoreEncoder(InputEncoder):
    def __init__(self, dtype=torch.float):
        super().__init__()
        self.dtype = dtype

    def encode(self, example):
        casted = float(example)
        return torch.tensor(casted, dtype=self.dtype)

    def batch_encode(self, iterator):
        return torch.tensor([self.encode(example) for example in iterator])


class BinaryScoreEncoder(ScoreEncoder):
    """Transform HTER score into binary OK/BAD label."""

    def encode(self, example):
        return super().encode(example).ceil().long()


class AlignmentEncoder(InputEncoder):
    def __init__(
        self, dtype=torch.int, account_for_bos_token=True, account_for_eos_token=True
    ):
        super().__init__()
        self.tokenize = align_tokenize
        self.dtype = dtype
        self.account_for_bos_token = account_for_bos_token
        self.account_for_eos_token = account_for_eos_token

    def encode(self, example):
        src_mt_alignments = self.tokenize(example)
        matrix = np.array(src_mt_alignments)
        if self.account_for_bos_token:
            matrix += 1
            matrix = np.vstack(([0, 0], matrix))
        shape = np.max(matrix, axis=0) + 1
        indicator_matrix = np.zeros(shape)
        indicator_matrix[tuple(matrix.T)] = 1

        return indicator_matrix

    def batch_encode(self, iterator):
        batch = [self.encode(example) for example in iterator]
        shapes = np.array([matrix.shape for matrix in batch])
        max_shape = np.max(shapes, axis=0) + 1
        batch = np.array(
            [
                np.pad(
                    matrix,
                    pad_width=np.array([[0, 0], max_shape - matrix.shape]).T,
                    mode='constant',
                )
                for matrix in batch
            ]
        )
        return torch.tensor(batch, dtype=self.dtype)
