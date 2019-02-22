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

import logging
from functools import partial

import torch
from torchtext.vocab import Vectors

from kiwi.constants import PAD, START, STOP, UNK

logger = logging.getLogger(__name__)


class WordEmbeddings(Vectors):
    def __init__(
        self,
        name,
        emb_format='polyglot',
        binary=True,
        map_fn=lambda x: x,
        **kwargs
    ):
        """
        Arguments:
           emb_format: the saved embedding model format, choices are:
                       polyglot, word2vec, fasttext, glove and text
           binary: only for word2vec, fasttext and text
           map_fn: a function that maps special original tokens
                       to Polyglot tokens (e.g. <eos> to </S>)
           save_vectors: save a vectors cache
        """
        self.binary = binary
        self.emb_format = emb_format

        self.itos = None
        self.stoi = None
        self.dim = None
        self.vectors = None

        self.map_fn = map_fn
        super().__init__(name, **kwargs)

    def __getitem__(self, token):
        if token in self.stoi:
            token = self.map_fn(token)
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def cache(self, name, cache, url=None, max_vectors=None):
        if self.emb_format in ['polyglot', 'glove']:
            try:
                from polyglot.mapping import Embedding
            except ImportError:
                logger.error('Please install `polyglot` package first.')
                return None
            if self.emb_format == 'polyglot':
                embeddings = Embedding.load(name)
            else:
                embeddings = Embedding.from_glove(name)
            self.itos = embeddings.vocabulary.id_word
            self.stoi = embeddings.vocabulary.word_id
            self.dim = embeddings.shape[1]
            self.vectors = torch.Tensor(embeddings.vectors).view(-1, self.dim)

        elif self.emb_format in ['word2vec', 'fasttext']:
            try:
                from gensim.models import KeyedVectors
            except ImportError:
                logger.error('Please install `gensim` package first.')
                return None
            embeddings = KeyedVectors.load_word2vec_format(
                name, unicode_errors='ignore', binary=self.binary
            )
            self.itos = embeddings.index2word
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
            self.dim = embeddings.vector_size
            self.vectors = torch.Tensor(embeddings.vectors).view(-1, self.dim)

        elif self.emb_format == 'text':
            tokens = []
            vectors = []
            if self.binary:
                import pickle

                # vectors should be a dict mapping str keys to numpy arrays
                with open(name, 'rb') as f:
                    d = pickle.load(f)
                    tokens = list(d.keys())
                    vectors = list(d.values())
            else:
                # each line should contain a token and its following fields
                # <token> <vector_value_1> ... <vector_value_n>
                with open(name, 'r', encoding='utf8') as f:
                    for line in f:
                        if line:  # ignore empty lines
                            fields = line.rstrip().split()
                            tokens.append(fields[0])
                            vectors.append(list(map(float, fields[1:])))
            self.itos = tokens
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
            self.vectors = torch.Tensor(vectors)
            self.dim = self.vectors.shape[1]


def map_to_polyglot(token):
    mapping = {UNK: '<UNK>', PAD: '<PAD>', START: '<S>', STOP: '</S>'}
    if token in mapping:
        return mapping[token]
    return token


Polyglot = partial(
    WordEmbeddings, emb_format='polyglot', map_fn=map_to_polyglot
)
Word2Vec = partial(WordEmbeddings, emb_format='word2vec')
FastText = partial(WordEmbeddings, emb_format='fasttext')
Glove = partial(WordEmbeddings, emb_format='glove')
TextVectors = partial(WordEmbeddings, emb_format='text')

AvailableVectors = {
    'polyglot': Polyglot,
    'word2vec': Word2Vec,
    'fasttext': FastText,
    'glove': Glove,
    'text': TextVectors,
}
