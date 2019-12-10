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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from kiwi import constants as const
from kiwi import load_model
from kiwi.data import utils
from kiwi.data.builders import build_test_dataset
from kiwi.data.iterators import build_bucket_iterator
from kiwi.data.utils import cross_split_dataset, save_predicted_probabilities
from kiwi.lib import train
from kiwi.lib.utils import merge_namespaces
from kiwi.loggers import tracking_logger

logger = logging.getLogger(__name__)


def run_from_options(options):
    if options is None:
        return

    meta_options = options.meta
    pipeline_options = options.pipeline.pipeline
    model_options = options.pipeline.model
    ModelClass = options.pipeline.model_api

    tracking_run = tracking_logger.configure(
        run_uuid=pipeline_options.run_uuid,
        experiment_name=pipeline_options.experiment_name,
        run_name=pipeline_options.run_name,
        tracking_uri=pipeline_options.mlflow_tracking_uri,
        always_log_artifacts=pipeline_options.mlflow_always_log_artifacts,
    )

    with tracking_run:
        output_dir = train.setup(
            output_dir=pipeline_options.output_dir,
            debug=pipeline_options.debug,
            quiet=pipeline_options.quiet,
        )

        all_options = merge_namespaces(
            meta_options, pipeline_options, model_options
        )
        train.log(
            output_dir,
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
    train_set, dev_set = train.retrieve_datasets(
        fieldset, pipeline_options, model_options, output_dir
    )

    test_set = None
    try:
        test_set = build_test_dataset(fieldset, **vars(pipeline_options))
    except ValueError:
        pass
    except FileNotFoundError:
        pass

    device_id = None
    if pipeline_options.gpu_id is not None and pipeline_options.gpu_id >= 0:
        device_id = pipeline_options.gpu_id

    parent_dir = output_dir
    train_predictions = defaultdict(list)
    dev_predictions = defaultdict(list)
    test_predictions = defaultdict(list)
    splitted_datasets = cross_split_dataset(train_set, splits)
    for i, (train_fold, pred_fold) in enumerate(splitted_datasets):

        run_name = 'train_split_{}'.format(i)
        output_dir = Path(parent_dir, run_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        # options.output_dir = str(options.output_dir)

        # Train
        vocabs = utils.fields_to_vocabs(train_fold.fields)

        tracking_run = tracking_logger.start_nested_run(run_name=run_name)
        with tracking_run:
            train.setup(
                output_dir=output_dir,
                seed=pipeline_options.seed,
                gpu_id=pipeline_options.gpu_id,
                debug=pipeline_options.debug,
                quiet=pipeline_options.quiet,
            )

            trainer = train.retrieve_trainer(
                ModelClass,
                pipeline_options,
                model_options,
                vocabs,
                output_dir,
                device_id,
            )

            # Dataset iterators
            train_iter = build_bucket_iterator(
                train_fold,
                batch_size=pipeline_options.train_batch_size,
                is_train=True,
                device=device_id,
            )
            valid_iter = build_bucket_iterator(
                pred_fold,
                batch_size=pipeline_options.valid_batch_size,
                is_train=False,
                device=device_id,
            )

            trainer.run(train_iter, valid_iter, epochs=pipeline_options.epochs)

        # Predict
        predictor = load_model(trainer.checkpointer.best_model_path())
        train_predictions_i = predictor.run(
            pred_fold, batch_size=pipeline_options.valid_batch_size
        )

        dev_predictions_i = predictor.run(
            dev_set, batch_size=pipeline_options.valid_batch_size
        )

        test_predictions_i = None
        if test_set:
            test_predictions_i = predictor.run(
                test_set, batch_size=pipeline_options.valid_batch_size
            )

        torch.cuda.empty_cache()

        for output_name in train_predictions_i:
            train_predictions[output_name] += train_predictions_i[output_name]
            dev_predictions[output_name].append(dev_predictions_i[output_name])
            if test_set:
                test_predictions[output_name].append(
                    test_predictions_i[output_name]
                )

    dev_predictions = average_all(dev_predictions)
    if test_set:
        test_predictions = average_all(test_predictions)

    save_predicted_probabilities(
        parent_dir, train_predictions, prefix=const.TRAIN
    )
    save_predicted_probabilities(parent_dir, dev_predictions, prefix=const.DEV)
    if test_set:
        save_predicted_probabilities(
            parent_dir, test_predictions, prefix=const.TEST
        )

    teardown(pipeline_options)

    return train_predictions


def teardown(options):
    pass


def average_all(predictions):
    for output_name in predictions:
        predictions[output_name] = average_predictions(predictions[output_name])
    return predictions


def average_predictions(ensemble):
    """Average an ensemble of predictions.
    """
    word_level = isinstance(ensemble[0][0], list)
    if word_level:
        sentence_lengths = [len(sentence) for sentence in ensemble[0]]
        ensemble = [
            [word for sentence in predictions for word in sentence]
            for predictions in ensemble
        ]

    ensemble = np.array(ensemble, dtype='float32')
    averaged_predictions = ensemble.mean(axis=0).tolist()

    if word_level:
        averaged_predictions = reshape_by_lengths(
            averaged_predictions, sentence_lengths
        )

    return averaged_predictions


def reshape_by_lengths(sequence, lengths):
    new_sequences = []
    t = 0
    for length in lengths:
        new_sequences.append(sequence[t : t + length])
        t += length
    return new_sequences
