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
from torchtext import data


def build_bucket_iterator(dataset, device, batch_size, is_train):
    device_obj = None if device is None else torch.device(device)
    iterator = data.BucketIterator(
        dataset=dataset,
        batch_size=batch_size,
        repeat=False,
        sort_key=dataset.sort_key,
        sort=False,
        # sorts the data within each minibatch in decreasing order
        # set to true if you want use pack_padded_sequences
        sort_within_batch=is_train,
        # shuffle batches
        shuffle=is_train,
        device=device_obj,
        train=is_train,
    )
    return iterator
