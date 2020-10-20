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
import math

from torch import nn

from kiwi.data.batch import BatchedSentence
from kiwi.modules.common.layer_norm import TFLayerNorm
from kiwi.modules.common.positional_encoding import PositionalEncoding
from kiwi.utils.io import BaseConfig


class TokenEmbeddings(nn.Module):
    class Config(BaseConfig):
        dim: int = 50
        freeze: bool = False
        dropout: float = 0.0
        use_position_embeddings: bool = False
        max_position_embeddings: int = 4000
        sparse_embeddings: bool = False
        scale_embeddings: bool = False
        input_layer_norm: bool = False

    def __init__(self, num_embeddings: int, pad_idx: int, config: Config, vectors=None):
        """A model for embedding a single type of tokens."""
        super().__init__()
        self.pad_idx = pad_idx

        if vectors is not None:
            assert num_embeddings == vectors.size(0)

            self.embedding = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=config.dim,
                padding_idx=pad_idx,
                sparse=config.sparse_embeddings,
                _weight=vectors,
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=config.dim,
                padding_idx=pad_idx,
                sparse=config.sparse_embeddings,
            )
            nn.init.xavier_uniform_(self.embedding.weight)

        self._size = config.dim
        self._pe = config.max_position_embeddings

        if config.freeze:
            self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(config.dropout)

        self.embeddings_scale_factor = 1
        if config.scale_embeddings:
            self.embeddings_scale_factor = math.sqrt(self._size)

        self.positional_encoding = None
        if config.use_position_embeddings:
            self.positional_encoding = PositionalEncoding(self._pe, self._size)

        self.layer_norm = None
        if config.input_layer_norm:
            self.layer_norm = TFLayerNorm(self._size)

    @property
    def num_embeddings(self):
        return self.embedding.num_embeddings

    def size(self):
        return self._size

    def forward(self, batch_input, *args):
        assert isinstance(batch_input, BatchedSentence)
        ids = batch_input.tensor

        embeddings = self.embedding(ids)
        embeddings = self.embeddings_scale_factor * embeddings

        if self.positional_encoding is not None:
            embeddings = self.positional_encoding(embeddings)

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)

        embeddings = self.dropout(embeddings)

        return embeddings
