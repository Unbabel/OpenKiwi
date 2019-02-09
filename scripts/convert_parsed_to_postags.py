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

import os


def read_file(fname):
    examples = []
    one_sample = []
    with open(fname, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                tok_line = line.split()
                if tok_line[0] == '1':
                    examples.append(one_sample)
                    one_sample = []
                one_sample.append(tok_line)
        examples.append(one_sample)
    return examples


def convert_to_pos(examples):
    new_examples = []
    for sample in examples:
        pos = []
        for sentence in sample:
            pos.append(sentence[3])
        new_examples.append(pos)
    return new_examples


def save_new_file(fname, pos_examples):
    print('Saving %s...' % fname)
    f = open(fname, 'w', encoding='utf8')
    for pos in pos_examples:
        s = ' '.join(pos)
        if s:
            f.write(s + '\n')
    f.close()


if __name__ == '__main__':
    for dname in os.listdir('.'):
        if not os.path.isdir(dname):
            continue
        prefix = 'train'
        if 'test' in dname:
            # prefix = 'test.2017'
            prefix = 'test'
        elif 'dev' in dname:
            prefix = 'dev'
        if ('training' in dname and 'dev' in dname) or ('.py' in dname):
            continue
        try:
            source_parsed = os.path.join(dname, prefix + '.src.parsed')
            target_parsed = os.path.join(dname, prefix + '.mt.parsed')
            examples_source = read_file(source_parsed)
            examples_target = read_file(target_parsed)
            pos_source = convert_to_pos(examples_source)
            pos_target = convert_to_pos(examples_target)
            save_new_file(os.path.join(dname, prefix + '.src.pos'), pos_source)
            save_new_file(os.path.join(dname, prefix + '.mt.pos'), pos_target)
        except FileNotFoundError as e:
            print('Skipping dir; parsed file not found: {}'.format(e))
            continue
