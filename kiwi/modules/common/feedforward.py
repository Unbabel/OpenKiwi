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

from torch import nn


def feedforward(
    in_dim,
    n_layers,
    shrink=2,
    out_dim=None,
    activation=nn.Tanh,
    final_activation=False,
    dropout=0.0,
):
    """Constructor for FeedForward Layers"""
    dim = in_dim
    module_dict = OrderedDict()
    for layer_i in range(n_layers - 1):
        next_dim = dim // shrink
        module_dict['linear_{}'.format(layer_i)] = nn.Linear(dim, next_dim)
        module_dict['activation_{}'.format(layer_i)] = activation()
        module_dict['dropout_{}'.format(layer_i)] = nn.Dropout(dropout)
        dim = next_dim
    next_dim = out_dim or (dim // 2)
    module_dict['linear_{}'.format(n_layers - 1)] = nn.Linear(dim, next_dim)
    if final_activation:
        module_dict['activation_{}'.format(n_layers - 1)] = activation()
    return nn.Sequential(module_dict)
