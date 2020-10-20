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
from math import sqrt

import torch
from torch import nn

from kiwi.utils.tensors import make_mergeable_tensors


class Scorer(nn.Module):
    """Score function for attention module.

    Arguments:
        scaled: whether to scale scores by `sqrt(hidden_size)` as proposed by the
                "Attention is All You Need" paper.
    """

    def __init__(self, scaled: bool = True):
        super().__init__()
        self.scaled = scaled

    def scale(self, hidden_size: int) -> float:
        """Denominator for scaling the scores.

        Arguments:
            hidden_size: max hidden size between query and keys.

        Return:
            sqrt(hidden_size) if `scaled` is True, 1 otherwise.
        """
        if self.scaled:
            return sqrt(hidden_size)
        return 1

    def forward(
        self, query: torch.FloatTensor, keys: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Compute scores for each key of size n given the queries of size m.

        The three dots (...) represent any other dimensions, such as the
        number of heads (useful if you use a multi head attention).

        Arguments:
            query: query matrix ``(bs, ..., target_len, m)``.
            keys: keys matrix ``(bs, ..., source_len, n)``.

        Return:
             matrix representing scores between source words and target words
             ``(bs, ..., target_len, source_len)``
        """
        raise NotImplementedError


class MLPScorer(Scorer):
    """MultiLayerPerceptron Scorer with variable nb of layers and neurons."""

    def __init__(
        self, query_size, key_size, layer_sizes=None, activation=nn.Tanh, **kwargs
    ):
        super().__init__(**kwargs)
        if layer_sizes is None:
            layer_sizes = [(query_size + key_size) // 2]
        input_size = query_size + key_size  # concatenate query and keys
        output_size = 1  # produce a scalar for each alignment
        layer_sizes = [input_size] + layer_sizes + [output_size]
        sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        layers = []
        for n_in, n_out in sizes:
            layers.append(nn.Sequential(nn.Linear(n_in, n_out), activation()))
            # layers.append(nn.Linear(n_in, n_out))
            # layers.append(activation())
        self.layers = nn.ModuleList(layers)

    def forward(self, query, keys):
        x_query, x_keys = make_mergeable_tensors(query, keys)
        x = torch.cat((x_query, x_keys), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)  # remove last dimension
