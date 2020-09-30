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

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-pred', help='Target predictions', type=str)
    parser.add_argument('--gaps-pred', help='Gaps predictions', type=str)
    parser.add_argument('--output', help='Path to output', type=str)
    return parser.parse_args()


def main(args):
    output_file_path = Path(args.output)
    if output_file_path.exists() and output_file_path.is_dir():
        output_file_path = Path(output_file_path, 'predicted.prob')
        print('Output is a directory, saving to: {}'.format(output_file_path))
    elif not output_file_path.exists():
        if not output_file_path.parent.exists():
            output_file_path.parent.mkdir(parents=True)
    f = output_file_path.open('w', encoding='utf8')

    with open(args.target_pred) as f_target, open(args.gaps_pred) as f_gaps:
        for line_target, line_gaps in zip(f_target, f_gaps):
            try:
                # labels are probs
                pred_target = list(map(float, line_target.split()))
                pred_gaps = list(map(float, line_gaps.split()))
            except ValueError:
                # labels are <BAD> and <OK> tags
                pred_target = line_target.split()
                pred_gaps = line_gaps.split()
            new_preds = []
            for i in range(len(pred_gaps)):
                new_preds.append(str(pred_gaps[i]))
                if i < len(pred_target):
                    new_preds.append(str(pred_target[i]))
            f.write(' '.join(new_preds) + '\n')
    f.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
