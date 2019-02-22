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

import torch
from torch import nn


class Scorer(nn.Module):
    """Score function for Attention module.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query, keys):
        """Computes Scores for each key given the query.
           args:
                 query:  FloatTensor batch x n
                 keys:   FloatTensor batch x seq_length x m
           ret:
                 scores: FloatTensor batch x seq_length
        """
        raise NotImplementedError


class MLPScorer(Scorer):
    """Implements a score function based on a Multilayer Perceptron.
    """

    def __init__(self, query_size, key_size, layers=2, nonlinearity=nn.Tanh):
        super().__init__()
        layer_list = []
        size = query_size + key_size
        for i in range(layers):
            size_next = size // 2 if i < layers - 1 else 1
            layer_list.append(
                nn.Sequential(nn.Linear(size, size_next), nonlinearity())
            )
            size = size_next
        self.layers = nn.ModuleList(layer_list)

    def forward(self, query, keys):
        layer_in = torch.cat([query.unsqueeze(1).expand_as(keys), keys], dim=-1)
        layer_in = layer_in.reshape(-1, layer_in.size(-1))
        for layer in self.layers:
            layer_in = layer(layer_in)
        out = layer_in.reshape(keys.size()[:-1])
        return out
