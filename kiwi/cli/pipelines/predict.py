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

from kiwi.cli.better_argparse import PipelineParser
from kiwi.cli.models import linear, nuqe, predictor_estimator, quetch
from kiwi.lib import predict

logger = logging.getLogger(__name__)


def predict_opts(parser):
    group = parser.add_argument_group("predicting")
    group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Maximum batch size for predicting.",
    )


def build_parser():
    return PipelineParser(
        name="predict",
        model_parsers=[
            nuqe.parser_for_pipeline("predict"),
            predictor_estimator.parser_for_pipeline("predict"),
            quetch.parser_for_pipeline("predict"),
            linear.parser_for_pipeline("predict"),
        ],
        options_fn=predict_opts,
    )


def main(argv=None):
    parser = build_parser()
    options = parser.parse(args=argv)
    # is this needed?
    if options is None:
        return
    predict.predict_from_options(options)


if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover
