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

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog='Jackknifing')
    parser.add_argument(
        '--n-folds', help='Folds for Jackknifing', type=int, default=1
    )
    parser.add_argument(
        '--data',
        help='Path to data. postfixes (pe, src, mt) assumed.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--out-dir', help='Path to output dir', type=str, required=True
    )

    return parser.parse_args()


SRC, MT, PE = 'src', 'mt', 'pe'


def readfile(path):
    with open(path, 'r') as f:
        lines = [l.strip() + '\n' for l in f]
    return lines


def writefile(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(line)


def get_folds(data, n_folds):
    fold_size = len(data) // n_folds

    folds = [
        data[i * fold_size : (i + 1) * fold_size] for i in range(n_folds - 1)
    ]
    folds.append(data[(n_folds - 1) * fold_size :])
    return folds


def jackknife(folds):
    jackknifed = [folds[:i] + folds[i + 1 :] for i in range(len(folds))]
    jackknifed = [[x for fold in folds for x in fold] for folds in jackknifed]
    return jackknifed


def permute(src, pe):
    permutation = np.random.permutation(len(src))
    src = [src[pos] for pos in permutation]
    pe = [pe[pos] for pos in permutation]
    return src, pe


def main(args):
    src = readfile('{}.{}'.format(args.data, SRC))
    pe = readfile('{}.{}'.format(args.data, PE))
    src_folds = get_folds(src, args.n_folds)
    pe_folds = get_folds(pe, args.n_folds)
    src_jk = jackknife(src_folds)
    pe_jk = jackknife(pe_folds)
    data_name = Path(args.data).name
    for i, (src, pe, src_pred) in enumerate(zip(src_jk, pe_jk, src_folds)):
        if len(src) != len(pe):
            raise Exception('Source and PE have different sizes.')
        writefile(
            src_pred, '{}/fold_{}/pred_fold.src'.format(args.out_dir, i + 1)
        )
        writefile(
            src, '{}/fold_{}/{}.{}'.format(args.out_dir, i + 1, data_name, SRC)
        )
        writefile(
            pe, '{}/fold_{}/{}.{}'.format(args.out_dir, i + 1, data_name, PE)
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
