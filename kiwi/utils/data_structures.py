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

from kiwi import constants as const


class DefaultFrozenDict(OrderedDict):
    def __init__(self, mapping=None, default_key=const.UNK):
        if mapping is None:
            super().__init__()
        else:
            super().__init__(mapping)
        self._default_key = default_key

    def __getitem__(self, k):
        default_id = self.get(self._default_key)
        item = self.get(k, default_id)
        if item is None:
            raise KeyError(
                f"'{k}' (and default '{self._default_key}' not found either)"
            )
        return item
