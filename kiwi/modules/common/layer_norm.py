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
import torch
from torch import nn


class LayerNorm(nn.Module):
    """Construct a layer normalization module.

    It normalizes the outputs of neurons for a given layer:
    :math:`out = (\\gamma * (x - x.mean(-1)) / (x.std(-1) + \\epsilon)) + \\beta`

    References:
        https://arxiv.org/abs/1607.06450

    Arguments:
        hidden_size: number of neurons in the layer x.
        eps: factor to prevent division by zero.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class TFLayerNorm(nn.Module):
    """Construct a layer normalization module with epsilon inside the
    square root (tensorflow style).

    This is equivalent to HuggingFace's BertLayerNorm module.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.gamma * x + self.beta
