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


def extract_columns(filepath, columns, separator='\t'):
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            if line:
                fields = line.split(separator)
                print(separator.join([fields[i - 1] for i in columns]))
            else:
                print()


if __name__ == '__main__':
    import sys

    filepath = sys.argv[1]
    columns = [int(c) for c in sys.argv[2:]]
    extract_columns(filepath, columns)
