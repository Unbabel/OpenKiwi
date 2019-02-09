import logging

from kiwi.cli.better_argparse import PipelineParser
from kiwi.cli.models import linear, nuqe, predictor_estimator, quetch
from kiwi.lib import predict

logger = logging.getLogger(__name__)


def predict_opts(parser):
    group = parser.add_argument_group('predicting')
    group.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Maximum batch size for predicting.',
    )


def main(argv=None):
    parser = PipelineParser(
        name='predict',
        model_parsers=[
            nuqe.parser_for_pipeline('predict'),
            predictor_estimator.parser_for_pipeline('predict'),
            quetch.parser_for_pipeline('predict'),
            linear.parser_for_pipeline('predict'),
        ],
        options_fn=predict_opts,
    )
    options = parser.parse(args=argv)
    if options is None:
        return

    output_dir = predict.setup(options.pipeline)

    predict.run(options.model_api, output_dir, options.pipeline, options.model)
    predict.teardown(options.pipeline)


if __name__ == '__main__':  # pragma: no cover
    main()  # pragma: no cover
