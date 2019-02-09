import logging
from pathlib import Path
from pprint import pformat

from kiwi.data.builders import build_test_dataset
from kiwi.data.utils import (
    deserialize_fields_from_vocabs,
    save_predicted_probabilities,
)
from kiwi.lib.utils import (
    configure_device,
    configure_logging,
    configure_seed,
    save_config_file,
    setup_output_directory,
)
from kiwi.models.linear_word_qe_classifier import LinearWordQEClassifier
from kiwi.models.model import Model
from kiwi.predictors.linear_tester import LinearTester
from kiwi.predictors.predictor import Predicter

logger = logging.getLogger(__name__)


def load_model(model_path):
    """Load a pretrained model into a `Predicter` object.

    Args:
      load_model (str): A path to the saved model file.

    Throws:
      Exception: If the path does not exist, or is not a valid model file.

    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise Exception('Path "{}" does not exist!'.format(model_path))

    model = Model.create_from_file(model_path)
    if not model:
        raise Exception('No model found in "{}"'.format(model_path))
    fieldset = model.fieldset()
    fields = deserialize_fields_from_vocabs(fieldset.fields, model.vocabs)
    predicter = Predicter(model, fields=fields)
    return predicter


def run(ModelClass, output_dir, pipeline_opts, model_opts):
    model_name = getattr(ModelClass, 'title', ModelClass.__name__)
    logger.info('Predict with the {} model'.format(model_name))

    if ModelClass == LinearWordQEClassifier:
        load_vocab = None

        model = LinearWordQEClassifier(
            evaluation_metric=model_opts.evaluation_metric
        )
        model.load(pipeline_opts.load_model)
        predicter = LinearTester(model)
    else:
        load_vocab = pipeline_opts.load_model

        model = Model.create_from_file(pipeline_opts.load_model)

        # Set GPU or CPU. This has to be done before instantiating the optimizer
        device_id = None
        if pipeline_opts.gpu_id is not None and pipeline_opts.gpu_id >= 0:
            device_id = pipeline_opts.gpu_id
        model.to(device_id)

        predicter = Predicter(model)

    test_dataset = build_test_dataset(
        fieldset=ModelClass.fieldset(
            wmt18_format=model_opts.__dict__.get('wmt18_format')
        ),
        load_vocab=load_vocab,
        **vars(model_opts)
    )
    predictions = predicter.run(
        test_dataset, batch_size=pipeline_opts.batch_size
    )

    save_predicted_probabilities(output_dir, predictions)
    return predictions


def setup(options):
    output_dir = setup_output_directory(
        options.output_dir, options.run_uuid, experiment_id=None, create=True
    )
    configure_logging(
        output_dir=output_dir, debug=options.debug, quiet=options.quiet
    )
    configure_seed(options.seed)
    configure_device(options.gpu_id)

    logger.info(pformat(vars(options)))
    logger.info('Local output directory is: {}'.format(output_dir))

    if options.save_config:
        save_config_file(options, options.save_config)

    del options.output_dir  # FIXME: remove this after making sure no other
    # place uses it! # noqa
    return output_dir


def teardown(options):
    pass
