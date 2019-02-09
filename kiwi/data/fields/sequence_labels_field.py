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

from collections import Counter

from torchtext.data import Field


class SequenceLabelsField(Field):
    """Sequence of Labels.
    """

    def __init__(self, classes, *args, **kwargs):
        self.classes = classes
        self.vocab = None
        super().__init__(*args, **kwargs)

    def build_vocab(self, *args, **kwargs):
        specials = self.classes + [
            self.pad_token,
            self.init_token,
            self.eos_token,
        ]
        self.vocab = self.vocab_cls(Counter(), specials=specials, **kwargs)
