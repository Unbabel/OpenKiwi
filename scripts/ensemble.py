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

import logging

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
    help='File where to save the result.',
)

parser.add_argument(
    'predicted_file',
    nargs='+',
    type=str,
    help='Path to probabilities for BAD tokens.',
)


def reshape_by_lengths(sequence, lengths):
    new_sequences = []
    t = 0
    for length in lengths:
        new_sequences.append(sequence[t : t + length])
        t += length
    return new_sequences


def average(probabilities):
    flat_probabilities = [list(flatten(probs)) for probs in probabilities]
    if not all_equal([len(p) for p in flat_probabilities]):
        logger.error('Number of probabilities do not match.')
        return None
    samples_lengths = [len(sample_probs) for sample_probs in probabilities[0]]
    probabilities = np.array(flat_probabilities, dtype='float32')
    averaged_probabilities = probabilities.mean(axis=0)
    averaged_probabilities = reshape_by_lengths(
        averaged_probabilities.tolist(), samples_lengths
    )

    return averaged_probabilities


def run(options):
    if len(options.predicted_file) < 2:
        logger.error('Please provide at least 2 predicted files.')
        exit(-1)
    probabilities = [read_file(file) for file in options.predicted_file]
    lengths = [len(prob) for prob in probabilities]
    if min(lengths) != max(lengths):
        lengths_str = [
            '{}: {}'.format(file, length)
            for file, length in zip(options.predicted_file, lengths)
        ]
        lengths_str = '\n'.join(lengths_str)
        logger.error(
            'Files contain different number of lines:\n{}'.format(lengths_str)
        )
        exit(-1)

    averages = average(probabilities)
    save_file(options.output, averages)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
