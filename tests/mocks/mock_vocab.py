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
class MockVocab:
    vectors = None

    def __init__(self, dictionary):
        self.stoi = dictionary
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.length = max(self.itos.keys()) + 1

    def token_to_id(self, token):
        if token in self.stoi:
            return self.stoi[token]
        else:
            raise KeyError

    def __len__(self):
        return self.length
