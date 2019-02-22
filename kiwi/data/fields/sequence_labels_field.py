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
import copy

from torchtext.data import Field
from kiwi.data.vocabulary import Vocabulary
from kiwi import constants as const


class SequenceLabelsField(Field):
    """Sequence of Labels.
    """

    def __init__(self, classes=None, vocab_cls=Vocabulary, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if classes is None:
            classes = const.LABELS
        self.classes = classes
        self.vocab = None
        self.vocab_cls = vocab_cls

    def build_vocab(self, *args, **kwargs):
        specials = copy.copy(self.classes)
        if self.pad_token:
            specials.append(self.pad_token)
        # cnt = Counter({class_name: 1 for class_name in self.classes})
        self.vocab = self.vocab_cls(Counter(), specials=specials, **kwargs)
