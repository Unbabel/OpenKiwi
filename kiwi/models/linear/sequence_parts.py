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


class SequenceUnigramPart(object):
    """A part for unigrams (a single label at a word position)."""

    def __init__(self, index, label):
        self.label = label
        self.index = index


class SequenceBigramPart(object):
    """A part for bigrams (two labels at consecutive words position).
    Necessary for the model to be sequential."""

    def __init__(self, index, label, previous_label):
        self.label = label
        self.previous_label = previous_label
        self.index = index
