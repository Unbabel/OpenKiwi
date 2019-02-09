import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog='Oversampling and Jackknifing')
    parser.add_argument(
        '--n-folds', help='Folds for Jackknifing', type=int, default=1
    )
    parser.add_argument(
        '--oversample',
        help='Number of times to oversample wmt data',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--roundtrip',
        help='Path to roundtrip data. postfixes (pe, src, mt) assumed.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--wmt',
        help='Path to wmt data. postfixes (pe, src, mt) assumed.',
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


def jackknife(folds, data, i, n):
    fold_i = folds[:i] + folds[i + 1 :]
    fold_i = [x for fold in fold_i for x in fold]
    return data + n * fold_i


def permute(src, pe):
    permutation = np.random.permutation(len(src))
    src = [src[pos] for pos in permutation]
    pe = [pe[pos] for pos in permutation]
    return src, pe


def main(args):
    roundtrip_src = readfile('{}.{}'.format(args.roundtrip, SRC))
    roundtrip_pe = readfile('{}.{}'.format(args.roundtrip, PE))
    wmt_src = readfile('{}.{}'.format(args.wmt, SRC))
    wmt_pe = readfile('{}.{}'.format(args.wmt, PE))
    wmt_src = get_folds(wmt_src, args.n_folds)
    wmt_pe = get_folds(wmt_pe, args.n_folds)
    out_name = '{}.{}'.format(
        args.roundtrip.split('/')[-1], args.wmt.split('/')[-1]
    )
    for i in range(args.n_folds):
        src = jackknife(wmt_src, roundtrip_src, i, args.oversample)
        pe = jackknife(wmt_pe, roundtrip_pe, i, args.oversample)
        if len(src) != len(pe):
            raise
        src, pe = permute(src, pe)
        writefile(
            wmt_src[i], '{}/fold_{}/pred_fold.src'.format(args.out_dir, i + 1)
        )
        writefile(
            src, '{}/fold_{}/{}.{}'.format(args.out_dir, i + 1, out_name, SRC)
        )
        writefile(
            pe, '{}/fold_{}/{}.{}'.format(args.out_dir, i + 1, out_name, PE)
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
