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
from dataclasses import dataclass

import torch
import torchnlp.utils
from torchnlp.encoders.text import BatchedSequences


@dataclass
class BatchedSentence:
    tensor: torch.Tensor
    lengths: torch.Tensor
    bounds: torch.Tensor
    bounds_lengths: torch.Tensor
    strict_masks: torch.Tensor
    number_of_tokens: torch.Tensor

    def pin_memory(self):
        self.tensor = self.tensor.pin_memory()
        self.lengths = self.lengths.pin_memory()
        self.bounds = self.bounds.pin_memory()
        self.bounds_lengths = self.bounds_lengths.pin_memory()
        self.strict_masks = self.strict_masks.pin_memory()
        self.number_of_tokens = self.number_of_tokens.pin_memory()
        return self

    def to(self, *args, **kwargs):
        self.tensor = self.tensor.to(*args, **kwargs)
        self.lengths = self.lengths.to(*args, **kwargs)
        self.bounds = self.bounds.to(*args, **kwargs)
        self.bounds_lengths = self.bounds_lengths.to(*args, **kwargs)
        self.strict_masks = self.strict_masks.to(*args, **kwargs)
        self.number_of_tokens = self.number_of_tokens.to(*args, **kwargs)
        return self


class MultiFieldBatch(dict):
    def __init__(self, batch: dict):
        super().__init__()
        self.update(batch)

    def pin_memory(self):
        for field, data in self.items():
            if isinstance(data, BatchedSequences):
                tensor = data.tensor.pin_memory()
                lengths = data.lengths.pin_memory()
                self[field] = BatchedSequences(tensor=tensor, lengths=lengths)
            else:
                self[field] = data.pin_memory()
        return self

    def to(self, *args, **kwargs):
        for field, data in self.items():
            self[field] = data.to(*args, **kwargs)
        return self


def tensors_to(tensors, *args, **kwargs):
    if isinstance(tensors, (MultiFieldBatch, BatchedSentence)):
        return tensors.to(*args, **kwargs)
    elif isinstance(tensors, dict):
        return {k: tensors_to(v, *args, **kwargs) for k, v in tensors.items()}
    else:
        return torchnlp.utils.tensors_to(tensors, *args, **kwargs)
