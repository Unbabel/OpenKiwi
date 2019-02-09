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
