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
