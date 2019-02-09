import logging
import random
from argparse import Namespace
from pathlib import Path

import configargparse
import numpy as np
from more_itertools import all_equal, flatten

from kiwi.data.utils import read_file, save_file

logger = logging.getLogger(__name__)

parser = configargparse.get_argument_parser(
    description='Quality Estimation toolkit: ensemble of predictions'
)

parser.add_argument(
    '-o',
    '--output',
    type=str,
    required=True,
    help='Directory where to save the resulting files.',
)

parser.add_argument(
    'directory', type=str, help='Path to WMT word level directory.'
)


def build_vocab(sentences):
    return np.unique(np.array(list(flatten(sentences)))).tolist()


def run(options):
    wmt = Path(options.directory)
    dataset_name = 'train'
    files = ['src', 'mt', 'tags', 'align']

    dataset = Namespace()
    noisy_dataset = Namespace()
    for ext in files:
        content = read_file(wmt / '{}.{}'.format(dataset_name, ext))
        setattr(dataset, ext, content)
        setattr(noisy_dataset, ext, [])

    mt_words = build_vocab(dataset.mt)

    set_percentage = 0.3

    nb_samples = len(dataset.src)
    sampled_indices = random.sample(
        range(nb_samples), k=int(nb_samples * set_percentage)
    )
    for i in sampled_indices:
        mt_sentence = dataset.mt[i]
        sampled_words = random.sample(
            range(len(mt_sentence)), k=int(len(mt_sentence) * 0.3)
        )
        for j in sampled_words:
            mt_sentence[j] = random.choice(mt_words)
            dataset.tags[i][j] = 'BAD'
        noisy_dataset.mt.append(mt_sentence)
        noisy_dataset.tags.append(dataset.tags[i])
        noisy_dataset.src.append(dataset.src[i])
        noisy_dataset.align.append(dataset.align[i])

    out = Path(options.output)
    for ext in files:
        content = getattr(dataset, ext)
        content += getattr(noisy_dataset, ext)
        save_file(out / '{}.noisy.{}'.format(dataset_name, ext), content)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
