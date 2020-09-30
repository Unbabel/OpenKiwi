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
import warnings

from kiwi.utils.data_structures import DefaultFrozenDict

logger = logging.getLogger(__name__)


class Vocabulary:
    """Define a vocabulary object that will be used to numericalize a field.

    Attributes:
        counter: A collections.Counter object holding the frequencies of tokens in the
                 data used to build the Vocab.
        stoi: A dictionary mapping token strings to numerical identifiers;
              NOTE: use :meth:`token_to_id` to do the conversion.
        itos: A list of token strings indexed by their numerical identifiers;
              NOTE: use :meth:`id_to_token` to do the conversion.
    """

    def __init__(
        self,
        counter,
        max_size=None,
        min_freq=1,
        unk_token=None,
        pad_token=None,
        bos_token=None,
        eos_token=None,
        specials=None,
        vectors=None,
        unk_init=None,
        vectors_cache=None,
        specials_first=True,
        rare_with_vectors=True,
        add_vectors_vocab=False,
    ):
        """Create a Vocabulary object from a collections.Counter.

        Arguments:
            counter: :class:`collections.Counter` object holding the frequencies of
                each value found in the data.
            max_size: the maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: the minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            unk_token: special unknown token.
            pad_token: special pad token.
            bos_token: special beginning of sentence token.
            eos_token: special end of sentence token.
            specials: the list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to :param:`unk_token`,
                :param:`pad_token`, :param:`bos_token`, and :param:`eos_token`.
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors.
            unk_init (callback): by default, initialize out-of-vocabulary word
                vectors to zero vectors; can be any function that takes in a
                Tensor and returns a Tensor of the same size;
                default: :meth:`torch.Tensor.zero_`.
            vectors_cache: directory for cached vectors; default: '.vector_cache'.
            specials_first: whether special tokens are prepended to rest of vocab
                            (else, they are appended).
            rare_with_vectors: if True and a vectors object is passed, then
                it will add words that appears less than min_freq but are in
                vectors vocabulary.
            add_vectors_vocab: by default, the vocabulary is built using only
                words from the provided datasets. If this flag is true, the
                vocabulary will add words that are not in the datasets but are
                in the vectors vocabulary (e.g. words from polyglot vectors).
        """
        self.specials = []
        self.unk_token = unk_token
        if self.unk_token:
            self.specials.append(self.unk_token)
        self.pad_token = pad_token
        if self.pad_token:
            self.specials.append(self.pad_token)
        self.bos_token = bos_token
        if self.bos_token:
            self.specials.append(self.bos_token)
        self.eos_token = eos_token
        if self.eos_token:
            self.specials.append(self.eos_token)
        if specials is None:
            specials = []
        self.specials.extend(specials)

        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list()
        if specials_first:
            self.itos = list(self.specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:
            del counter[tok]

        if max_size is not None:
            max_size += len(self.itos)

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

        if not specials_first:
            self.itos.extend(list(self.specials))

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

        # stoi is simply a reverse dict for itos
        self.stoi = DefaultFrozenDict(
            {tok: i for i, tok in enumerate(self.itos)}, default_key=self.unk_token
        )

        self.vectors = None
        if vectors is not None:
            # self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
            raise NotImplementedError('Loading pretrained embeddings is not supported')
        # else:
        #     assert unk_init is None and vectors_cache is None

    # def __eq__(self, other):
    #     if other is None:
    #         return False
    #     return super().__eq__(other)

    def token_to_id(self, token):
        token_id = self.stoi.get(token)
        if token_id is None:
            if self.unk_token is None:
                raise KeyError('Token not in vocabulary: {}'.format(token))

            token_id = self.stoi.get(self.unk_token)
            if token_id is None:
                raise KeyError(
                    'Token not in vocabulary nor expected default token: {} '
                    'and {}'.format(token, self.unk_token)
                )
        return token_id

    def id_to_token(self, idx):
        return self.itos[idx]

    @property
    def pad_id(self):
        if self.pad_token:
            return self.stoi[self.pad_token]
        return None

    @property
    def bos_id(self):
        if self.bos_token:
            return self.stoi[self.bos_token]
        return None

    @property
    def eos_id(self):
        if self.eos_token:
            return self.stoi[self.eos_token]
        return None

    def __len__(self):
        return len(self.itos)

    def net_length(self):
        return self.__len__() - len(self.specials)

    def max_size(self, max_size):
        """Limit the vocabulary size.

        The assumption here is that the vocabulary was created from a list of tokens
        sorted by descending frequency.
        """
        assert max_size >= 1
        init_size = len(self)
        if isinstance(self.itos, list):
            self.itos = self.itos[:max_size]
            self.stoi = {s: i for i, s in enumerate(self.itos)}
        else:
            self.itos = {i: s for i, s in self.itos.items() if i < max_size}
            self.stoi = {s: i for i, s in self.itos.items()}
        # self.counts = {k: v for k, v in self.counts.items() if k in self.stoi}
        # self.check_valid()
        logger.info(
            f"Maximum vocabulary size: {max_size:d}. "
            f"Dictionary size: {init_size:d} -> {len(self):d} "
            f"(removed {init_size - len(self):d} words)."
        )

    def __getstate__(self):
        # This method is called when you are going to pickle the class,
        # to know what to pickle.
        state = self.__dict__.copy()
        del state['vectors']

        return state

    def __setstate__(self, state):
        # This method is called when you are going to unpickle the class,
        # If you need some initialization after the unpickling you can do
        # it here.

        self.__dict__.update(state)
        # if 'stoi' not in state:
        #     # stoi is simply a reverse dict for itos
        #     if isinstance(self.itos, dict):
        #         # Special case for BERT vocabularies
        #         self.stoi = DefaultFrozenDict(
        #             zip(self.itos.values(), self.itos.keys()),
        #             default_key=self.unk_token,
        #         )
        #     else:
        #         self.stoi = DefaultFrozenDict(
        #             {tok: i for i, tok in enumerate(self.itos)},
        #             default_key=self.unk_token,
        #         )

        self.vectors = None
