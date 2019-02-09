from kiwi.cli.better_argparse import PipelineParser
from kiwi.cli.models import predictor_estimator
from kiwi.cli.pipelines.train import train_opts
from kiwi.lib import preprocess


def main(argv=None):
    parser = PipelineParser(
        name='preprocess',
        model_parsers=[predictor_estimator.parser_for_pipeline('preprocess')],
        options_fn=train_opts,
        add_general_options=False,
    )
    options = parser.parse(args=argv)
    if options is None:
        return

    output_dir = preprocess.setup(options.pipeline)
    preprocess.run(options.model_api, output_dir, options.all_options)
    preprocess.teardown(options.pipeline)


if __name__ == '__main__':  # pragma: no cover
    main()  # pragma: no cover
