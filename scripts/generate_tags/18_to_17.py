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

import sys
from pathlib import Path

from kiwi import constants

if __name__ == '__main__':
    tags_file = sys.argv[1]
    tags = []
    with open(tags_file) as t:
        tags = [line.strip().split() for line in t]
    gap_tags = [x[::2] for x in tags]
    target_tags = [x[1::2] for x in tags]
    with open(str(Path(tags_file).parent / constants.GAP_TAGS), 'w') as g:
        for line in gap_tags:
            g.write(' '.join(line) + '\n')
    with open(str(Path(tags_file).parent / constants.TARGET_TAGS), 'w') as t:
        for line in target_tags:
            t.write(' '.join(line) + '\n')
