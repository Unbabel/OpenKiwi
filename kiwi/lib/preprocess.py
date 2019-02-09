import logging

from kiwi.data.builders import build_training_datasets
from kiwi.data.utils import save_training_datasets

logger = logging.getLogger(__name__)


def run(model_api, output_dir, model_options):
    logging.info('Preprocessing and saving Dataset')
    logger.info('Building datasets...')

    datasets = build_training_datasets(
        fieldset=model_api.fieldset(wmt18_format=model_options.wmt18_format),
        load_vocab=None,
        **vars(model_options)
    )

    logger.info('Saving preprocessed datasets...')
    save_training_datasets(output_dir, *datasets)


def setup(options):
    logging.debug(vars(options))
    output_dir = options.output_dir
    return output_dir


def teardown(options):
    pass
