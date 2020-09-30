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
from collections import OrderedDict
from typing import Dict

import torch
from pydantic import confloat
from torch import nn

from kiwi import constants as const
from kiwi.data.batch import MultiFieldBatch
from kiwi.systems._meta_module import MetaModule
from kiwi.utils.io import BaseConfig


@MetaModule.register_subclass
class LinearDecoder(MetaModule):
    class Config(BaseConfig):
        hidden_size: int = 250
        'Size of hidden layer'

        dropout: confloat(ge=0.0, le=1.0) = 0.0

        bottleneck_size: int = 100

    def __init__(self, inputs_dims, config):
        super().__init__(config=config)

        self.features_dims = inputs_dims

        # Build Model #
        self.linear_outs = nn.ModuleDict()
        self._sizes = {}
        if const.TARGET in self.features_dims:
            self.linear_outs[const.TARGET] = nn.Sequential(
                nn.Linear(self.features_dims[const.TARGET], self.config.hidden_size),
                nn.Tanh(),
            )
            self._sizes[const.TARGET] = self.config.hidden_size
        if const.SOURCE in self.features_dims:
            self.linear_outs[const.SOURCE] = nn.Sequential(
                nn.Linear(self.features_dims[const.SOURCE], self.config.hidden_size),
                nn.Tanh(),
            )
            self._sizes[const.SOURCE] = self.config.hidden_size

        self.dropout = nn.Dropout(self.config.dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        if const.TARGET_SENTENCE in self.features_dims:
            linear_layers = [
                nn.Linear(
                    self.features_dims[const.TARGET_SENTENCE],
                    self.config.bottleneck_size,
                ),
                nn.Tanh(),
                nn.Dropout(self.config.dropout),
            ]
            linear_layers.extend(
                [
                    nn.Linear(self.config.bottleneck_size, self.config.hidden_size),
                    nn.Tanh(),
                    nn.Dropout(self.config.dropout),
                ]
            )
            self.linear_outs[const.TARGET_SENTENCE] = nn.Sequential(*linear_layers)
            self._sizes[const.TARGET_SENTENCE] = self.config.hidden_size

    def size(self, field=None):
        if field:
            return self._sizes[field]
        return self._sizes

    def forward(self, features: Dict[str, torch.Tensor], batch_inputs: MultiFieldBatch):
        output_features = OrderedDict()

        if const.TARGET in features:
            features_tensor = self.dropout(features[const.TARGET])
            output_features[const.TARGET] = self.linear_outs[const.TARGET](
                features_tensor
            )
        if const.SOURCE in features:
            features_tensor = self.dropout(features[const.SOURCE])
            output_features[const.SOURCE] = self.linear_outs[const.SOURCE](
                features_tensor
            )
        if const.TARGET_SENTENCE in features:
            features_tensor = self.linear_outs[const.TARGET_SENTENCE](
                features[const.TARGET_SENTENCE]
            )
            output_features[const.TARGET_SENTENCE] = features_tensor

        return output_features
