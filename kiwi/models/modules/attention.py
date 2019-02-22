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


class Attention(nn.Module):
    """Generic Attention Implementation.
       Module computes a convex combination of a set of values based on the fit
       of their keys with a query.
    """

    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer
        self.mask = None

    def forward(self, query, keys, values=None):
        if values is None:
            values = keys
        scores = self.scorer(query, keys)
        # Masked Softmax
        scores = scores - scores.mean(1, keepdim=True)  # numerical stability
        scores = torch.exp(scores)
        if self.mask is not None:
            scores = self.mask * scores
        convex = scores / scores.sum(1, keepdim=True)
        return torch.einsum('bs,bsi->bi', [convex, values])

    def set_mask(self, mask):
        self.mask = mask
