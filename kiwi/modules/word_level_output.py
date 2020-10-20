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


class WordLevelOutput(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        pad_idx,
        class_weights=None,
        remove_first=False,
        remove_last=False,
    ):
        super().__init__()

        self.pad_idx = pad_idx

        # Explicit check to avoid using 0 as False
        self.start_pos = None if remove_first is False or remove_first is None else 1
        self.stop_pos = None if remove_last is False or remove_last is None else -1

        self.linear = nn.Linear(input_size, output_size)

        self.loss_fn = nn.CrossEntropyLoss(
            reduction='sum', ignore_index=pad_idx, weight=class_weights
        )

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, features_tensor, batch_inputs=None):
        logits = self.linear(features_tensor)
        logits = logits[:, self.start_pos : self.stop_pos]
        return logits


class GapTagsOutput(WordLevelOutput):
    def __init__(
        self,
        input_size,
        output_size,
        pad_idx,
        class_weights=None,
        remove_first=False,
        remove_last=False,
    ):
        super().__init__(
            input_size=2 * input_size,
            output_size=output_size,
            pad_idx=pad_idx,
            class_weights=class_weights,
            remove_first=False,
            remove_last=False,
        )
        self.add_pad_start = 1 if remove_first is False or remove_first is None else 0
        self.add_pad_stop = 1 if remove_last is False or remove_last is None else 0

    def forward(self, features_tensor, batch_inputs=None):
        h_gaps = features_tensor
        if self.add_pad_start or self.add_pad_stop:
            # Pad dim=1
            num_of_pads = self.add_pad_start + self.add_pad_stop
            h_gaps = nn.functional.pad(
                h_gaps,
                pad=[0, 0] * (len(h_gaps.shape) - num_of_pads)
                + [self.add_pad_start, self.add_pad_stop],
                value=0,
            )
        h_gaps = torch.cat((h_gaps[:, :-1], h_gaps[:, 1:]), dim=-1)
        logits = super().forward(h_gaps, batch_inputs)
        return logits
