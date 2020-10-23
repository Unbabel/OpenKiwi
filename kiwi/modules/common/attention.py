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
from torch.nn import functional as F

from kiwi.utils.tensors import unsqueeze_as


class Attention(nn.Module):
    """Generic Attention Implementation.

       1. Use `query` and `keys` to compute scores (energies)
       2. Apply softmax to get attention probabilities
       3. Perform a dot product between `values` and probabilites (outputs)

    Arguments:
        scorer (kiwi.modules.common.Scorer): a scorer object
        dropout (float): dropout rate after softmax (default: 0.)
    """

    def __init__(self, scorer, dropout=0):
        super().__init__()
        self.scorer = scorer
        self.dropout = nn.Dropout(p=dropout)
        self.NEG_INF = -1e9  # for masking attention scores before softmax

    def forward(self, query, keys, values=None, mask=None):
        """Compute the attention between query, keys and values.

        Arguments:
            query (torch.Tensor): set of query vectors with shape of
                (batch_size, ..., target_len, hidden_size)
            keys (torch.Tensor): set of keys vectors with shape of
                (batch_size, ..., source_len, hidden_size)
            values (torch.Tensor, optional): set of values vectors with
                shape of: (batch_size, ..., source_len, hidden_size).
                If None, keys are treated as values. Default: None
            mask (torch.ByteTensor, optional): Tensor representing valid
                positions. If None, all positions are considered valid.
                Shape of (batch_size, target_len)

        Return:
            torch.Tensor: combination of values and attention probabilities.
                Shape of (batch_size, ..., target_len, hidden_size)
            torch.Tensor: attention probabilities between query and keys.
                Shape of (batch_size, ..., target_len, source_len)
        """
        if values is None:
            values = keys

        # get scores (aka energies)
        scores = self.scorer(query, keys)

        # mask out scores to infinity before softmax
        if mask is not None:
            # broadcast in keys' timestep dim many times as needed
            mask = unsqueeze_as(mask, scores, dim=-2)
            scores = scores.masked_fill(mask == 0, self.NEG_INF)

        # apply softmax to get probs
        p_attn = F.softmax(scores, dim=-1)

        # apply dropout - used in Transformer (default: 0)
        p_attn = self.dropout(p_attn)

        # dot product between p_attn and values
        # o_attn = torch.matmul(p_attn, values)
        o_attn = torch.einsum('b...ts,b...sm->b...tm', [p_attn, values])
        return o_attn, p_attn
