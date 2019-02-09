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


def tokenizer(sentence):
    """Implement your own tokenize procedure."""
    return sentence.strip().split()


def align_tokenizer(s):
    """Return a list of pair of integers for each sentence."""
    return [tuple(map(int, x.split('-'))) for x in s.strip().split()]


def align_reversed_tokenizer(s):
    """Return a list of pair of integers for each sentence."""
    return [tuple(map(int, x.split('-')))[::-1] for x in s.strip().split()]
