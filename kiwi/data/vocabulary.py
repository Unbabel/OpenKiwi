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

import warnings
from collections import defaultdict

import torchtext

from kiwi.constants import PAD, START, STOP, UNALIGNED, UNK, UNK_ID


def _default_unk_index():
    return UNK_ID  # should be zero


class Vocabulary(torchtext.vocab.Vocab):
    """Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(
        self,
        counter,
        max_size=None,
        min_freq=1,
        specials=None,
        vectors=None,
        unk_init=None,
        vectors_cache=None,
        rare_with_vectors=True,
        add_vectors_vocab=False,
    ):
        """Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word
                vectors to zero vectors; can be any function that takes in a
                Tensor and returns a Tensor of the same size.
                Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors.
                Default: '.vector_cache'
            rare_with_vectors: if True and a vectors object is passed, then
                it will add words that appears less than min_freq but are in
                vectors vocabulary. Default: True.
            add_vectors_vocab: by default, the vocabulary is built using only
                words from the provided datasets. If this flag is true, the
                vocabulary will add words that are not in the datasets but are
                in the vectors vocabulary (e.g. words from polyglot vectors).
                Default: False.
        """
        if specials is None:
            specials = ['<pad>']

        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        if not isinstance(vectors, list) and vectors is not None:
            vectors = [vectors]

        # add words that appears less than min_freq but are in embeddings
        # vocabulary
        for word, freq in words_and_frequencies:
            if freq < min_freq:
                if vectors is not None and rare_with_vectors:
                    for v in vectors:
                        if word in v.stoi:
                            self.itos.append(word)
                else:
                    break
            elif len(self.itos) == max_size:
                break
            else:
                self.itos.append(word)

        if add_vectors_vocab:
            if (
                max_size is not None
                and sum(v.stoi for v in vectors) + len(self.itos) > max_size
            ):
                warnings.warn(
                    'Adding the vectors vocabulary will make '
                    'len(vocab) > max_vocab_size!'
                )
            vset = set()
            for v in vectors:
                vset.update(v.stoi.keys())
            v_itos = vset - set(self.itos)
            self.itos.extend(list(v_itos))

        self.stoi = defaultdict(_default_unk_index)
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None


def merge_vocabularies(vocab_a, vocab_b, max_size=None, vectors=None, **kwargs):
    merged = vocab_a.freqs + vocab_b.freqs
    return Vocabulary(
        merged,
        specials=[UNK, PAD, START, STOP, UNALIGNED],
        max_size=max_size,
        vectors=vectors,
        **kwargs,
    )
