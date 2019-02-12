from kiwi.cli.better_argparse import ModelParser
from kiwi.cli.models.predictor_estimator import (
    add_pretraining_options
)
from kiwi.models.predictor import Predictor


def parser_for_pipeline(pipeline):
    if pipeline == 'train':
        return ModelParser(
            'predictor',
            'train',
            title=Predictor.title,
            options_fn=add_pretraining_options,
            api_module=Predictor,
        )
    return None
