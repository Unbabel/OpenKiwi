import logging
from collections import defaultdict
from pathlib import Path

import torch

from kiwi import load_model
from kiwi.data import utils
from kiwi.data.iterators import build_bucket_iterator
from kiwi.data.utils import cross_split_dataset, save_predicted_probabilities
from kiwi.lib.train import log, make_trainer, retrieve_datasets, setup
from kiwi.lib.utils import merge_namespaces
from kiwi.loggers import mlflow_logger

logger = logging.getLogger(__name__)


def run_from_options(options):
    if options is None:
        return

    meta_options = options.meta
    pipeline_options = options.pipeline.pipeline
    model_options = options.pipeline.model
    ModelClass = options.pipeline.model_api

    mlflow_run = mlflow_logger.configure(
        run_uuid=pipeline_options.run_uuid,
        experiment_name=pipeline_options.experiment_name,
        tracking_uri=pipeline_options.mlflow_tracking_uri,
        always_log_artifacts=pipeline_options.mlflow_always_log_artifacts,
    )

    with mlflow_run:
        output_dir = setup(
            output_dir=pipeline_options.output_dir,
            debug=pipeline_options.debug,
            quiet=pipeline_options.quiet,
        )

        all_options = merge_namespaces(
            meta_options, pipeline_options, model_options
        )
        log(
            output_dir,
            save_config=False,
            config_options=vars(all_options),
            config_file_name='jackknife_config.yml',
        )

        run(
            ModelClass,
            output_dir,
            pipeline_options,
            model_options,
            splits=meta_options.splits,
        )

    teardown(pipeline_options)


def run(ModelClass, output_dir, pipeline_options, model_options, splits):
    model_name = getattr(ModelClass, 'title', ModelClass.__name__)
    logger.info('Jackknifing with the {} model'.format(model_name))

    # Data
    fieldset = ModelClass.fieldset(
        wmt18_format=model_options.__dict__.get('wmt18_format')
    )
    train_dataset, _, *extra_datasets = retrieve_datasets(
        fieldset, pipeline_options, model_options, output_dir
    )

    device_id = None
    if pipeline_options.gpu_id is not None and pipeline_options.gpu_id >= 0:
        device_id = pipeline_options.gpu_id

    parent_dir = output_dir
    train_predictions = defaultdict(list)
    for i, (train_fold, dev_fold) in enumerate(
        cross_split_dataset(train_dataset, splits)
    ):
        run_name = 'train_split_{}'.format(i)
        output_dir = Path(parent_dir, run_name)
        output_dir.mkdir(parents=True, exist_ok=False)
        # options.output_dir = str(options.output_dir)

        # Trainer step
        vocabs = utils.fields_to_vocabs(train_fold.fields)
        model = ModelClass.from_options(vocabs=vocabs, opts=model_options)

        trainer = make_trainer(
            model, pipeline_options, model_options, output_dir, device_id
        )

        logger.info(str(trainer.model))
        logger.info('{} parameters'.format(trainer.model.num_parameters()))

        # Dataset iterators
        train_iter = build_bucket_iterator(
            train_fold,
            batch_size=pipeline_options.train_batch_size,
            is_train=True,
            device=device_id,
        )
        valid_iter = build_bucket_iterator(
            dev_fold,
            batch_size=pipeline_options.valid_batch_size,
            is_train=False,
            device=device_id,
        )

        # Train
        mlflow_run = mlflow_logger.start_nested_run(run_name=run_name)
        with mlflow_run:
            setup(
                output_dir=output_dir,
                seed=pipeline_options.seed,
                gpu_id=pipeline_options.gpu_id,
                debug=pipeline_options.debug,
                quiet=pipeline_options.quiet,
            )

            trainer.run(train_iter, valid_iter, epochs=pipeline_options.epochs)

        # Predict
        predictor = load_model(trainer.checkpointer.best_model_path())
        predictions = predictor.run(
            dev_fold, batch_size=pipeline_options.valid_batch_size
        )

        torch.cuda.empty_cache()

        for output_name, output_values in predictions.items():
            train_predictions[output_name] += output_values

    save_predicted_probabilities(parent_dir, train_predictions)

    teardown(pipeline_options)

    return train_predictions


def teardown(options):
    pass
