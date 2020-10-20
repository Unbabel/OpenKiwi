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
from pydantic import confloat, conlist
from torch import nn

from kiwi import constants as const
from kiwi.systems._meta_module import MetaModule
from kiwi.utils.io import BaseConfig


class NuQETargetDecoder(nn.Module):
    class Config(BaseConfig):
        hidden_sizes: conlist(int, min_items=4, max_items=4) = [400, 200, 100, 50]
        dropout: confloat(ge=0.0, le=1.0) = 0.4

    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        l1_dim = self.config.hidden_sizes[0]
        l2_dim = self.config.hidden_sizes[1]
        l3_dim = self.config.hidden_sizes[2]
        l4_dim = self.config.hidden_sizes[3]
        self._size = l4_dim

        # Linear layers
        self.linear_in = nn.Linear(input_dim, l1_dim)
        self.linear_2 = nn.Linear(l1_dim, l1_dim)
        self.linear_3 = nn.Linear(2 * l2_dim, l2_dim)
        self.linear_4 = nn.Linear(l2_dim, l2_dim)
        self.linear_5 = nn.Linear(2 * l2_dim, l3_dim)
        self.linear_6 = nn.Linear(l3_dim, l4_dim)

        # Recurrent Layers
        self.gru_1 = nn.GRU(l1_dim, l2_dim, bidirectional=True, batch_first=True)
        self.gru_2 = nn.GRU(l2_dim, l2_dim, bidirectional=True, batch_first=True)

        # Dropout after linear layers
        self.dropout_in = nn.Dropout(self.config.dropout)
        self.dropout_out = nn.Dropout(self.config.dropout)

        # Explicit initializations
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.xavier_uniform_(self.linear_5.weight)
        nn.init.xavier_uniform_(self.linear_6.weight)

        nn.init.constant_(self.linear_in.bias, 0.0)
        nn.init.constant_(self.linear_2.bias, 0.0)
        nn.init.constant_(self.linear_3.bias, 0.0)
        nn.init.constant_(self.linear_4.bias, 0.0)
        nn.init.constant_(self.linear_5.bias, 0.0)
        nn.init.constant_(self.linear_6.bias, 0.0)

        # Activation function
        self._activation_fn = torch.relu

    def forward(self, features, batch_inputs):
        h = self.dropout_in(features[const.TARGET])
        #
        # First linears
        #
        # (bs, ts, 2 * window * emb) -> (bs, ts, l1_dim)j
        h = self._activation_fn(self.linear_in(h))
        # (bs, ts, l1_dim) -> (bs, ts, l1_dim)
        h = self._activation_fn(self.linear_2(h))
        #
        # First recurrent
        #
        # (bs, ts, l1_dim) -> (bs, ts, l1_dim)
        h, _ = self.gru_1(h)
        #
        # Second linears
        #
        # (bs, ts, l1_dim) -> (bs, ts, l2_dim)
        h = self._activation_fn(self.linear_3(h))
        # (bs, ts, l2_dim) -> (bs, ts, l2_dim)
        h = self._activation_fn(self.linear_4(h))
        #
        # Second recurrent
        #
        # (bs, ts, l2_dim) -> (bs, ts, l2_dim)
        h, _ = self.gru_2(h)
        #
        # Third linears
        #
        # (bs, ts, l1_dim) -> (bs, ts, l3_dim)
        h = self._activation_fn(self.linear_5(h))
        # (bs, ts, l3_dim) -> (bs, ts, l4_dim)
        h = self._activation_fn(self.linear_6(h))
        h = self.dropout_out(h)
        return h

    def size(self):
        return self._size


class NuQESourceDecoder(nn.Module):
    class Config(BaseConfig):
        hidden_sizes: conlist(int, min_items=4, max_items=4) = [400, 200, 100, 50]
        dropout: confloat(ge=0.0, le=1.0) = 0.4

    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        l1_dim = self.config.hidden_sizes[0]
        l2_dim = self.config.hidden_sizes[1]
        l4_dim = self.config.hidden_sizes[3]
        self._size = l4_dim

        # Linear layers
        self.linear_source_target = nn.Linear(2 * l2_dim, l2_dim)
        self.linear_target_source = nn.Linear(2 * l2_dim, l2_dim)

        self.linear_source_in = nn.Linear(input_dim, l1_dim)
        self.linear_source_2 = nn.Linear(l1_dim, l1_dim)
        self.linear_source_3 = nn.Linear(2 * l2_dim, l2_dim)
        self.linear_source_6 = nn.Linear(l2_dim, l4_dim)
        self.gru_source_1 = nn.GRU(l1_dim, l2_dim, bidirectional=True, batch_first=True)

        nn.init.xavier_uniform_(self.linear_source_target.weight)
        nn.init.xavier_uniform_(self.linear_target_source.weight)
        nn.init.constant_(self.linear_source_target.bias, 0.0)
        nn.init.constant_(self.linear_target_source.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_source_in.weight)
        nn.init.xavier_uniform_(self.linear_source_2.weight)
        nn.init.xavier_uniform_(self.linear_source_3.weight)
        nn.init.xavier_uniform_(self.linear_source_6.weight)
        nn.init.constant_(self.linear_source_in.bias, 0.0)
        nn.init.constant_(self.linear_source_2.bias, 0.0)
        nn.init.constant_(self.linear_source_3.bias, 0.0)
        nn.init.constant_(self.linear_source_6.bias, 0.0)

        # Activation function
        self._activation_fn = torch.relu

    def forward(self, features, batch_inputs):
        h_source = features[const.SOURCE]

        h_source = self.linear_source_in(h_source)
        h_source = self._activation_fn(h_source)  # This improves

        h_source = self._activation_fn(self.linear_source_2(h_source))
        h_source, _ = self.gru_source_1(h_source)
        h_source = self._activation_fn(self.linear_source_3(h_source))

        h_source = self._activation_fn(self.linear_source_6(h_source))

        return h_source

    def size(self):
        return self._size


@MetaModule.register_subclass
class NuQEDecoder(MetaModule):
    """Neural Quality Estimation (NuQE) model for word level quality estimation."""

    class Config(BaseConfig):
        target: NuQETargetDecoder.Config
        source: NuQESourceDecoder.Config

    def __init__(self, inputs_dims, config: Config):
        super().__init__(config=config)

        self.features_dims = inputs_dims

        self.decoders = nn.ModuleDict()
        self.decoders[const.TARGET] = NuQETargetDecoder(
            input_dim=self.features_dims[const.TARGET], config=self.config.target
        )
        self.decoders[const.SOURCE] = NuQESourceDecoder(
            input_dim=self.features_dims[const.SOURCE], config=self.config.source
        )

        self._sizes = {
            const.TARGET: self.decoders[const.TARGET].size(),
            const.SOURCE: self.decoders[const.SOURCE].size(),
        }

    def forward(self, features, batch_inputs):
        target = self.decoders[const.TARGET](features, batch_inputs)
        source = self.decoders[const.SOURCE](features, batch_inputs)

        return {const.TARGET: target, const.SOURCE: source}

    def size(self, field=None):
        if field:
            return self._sizes[field]
        return self._sizes
