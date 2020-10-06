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
import datetime
import logging
from functools import partial
from pathlib import Path
from typing import List

import hydra
import joblib
import optuna
from optuna.samplers import TPESampler
from optuna.study import Study
from pydantic import FilePath
from pydantic.dataclasses import dataclass

import kiwi.cli
from kiwi.lib import train
from kiwi.lib.utils import (
    configure_logging,
    configure_seed,
    load_config,
    save_config_to_file,
)
from kiwi.utils.io import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class OutputFiles:
    directory: Path = 'optunaruns'


class LearningRateRange(BaseConfig):
    lower: float = 5e-7
    upper: float = 1e-5


class DropoutRange(BaseConfig):
    lower: float = 0.0
    upper: float = 0.25


class WarmupRange(BaseConfig):
    lower: float = 0.05
    upper: float = 0.25


class FreezeRange(BaseConfig):
    lower: int = 0
    upper: int = 6


class SentenceWeightRange(BaseConfig):
    lower: float = 1.0
    upper: float = 5.0


class SearchOptions(BaseConfig):
    learning_rate: LearningRateRange = LearningRateRange()
    dropout: DropoutRange = DropoutRange()
    warmup_steps: WarmupRange = WarmupRange()
    freeze_epochs: FreezeRange = FreezeRange()
    sentence_loss_weight: SentenceWeightRange = None
    patience: int = 10
    validation_steps: float = 0.2
    hidden_size: List[int] = None
    bottleneck_size: List[int] = None
    search_mlp: bool = False
    search_hter: bool = False
    search_word_level: bool = False
    search_sentence_loss_weight: bool = False


class Configuration(BaseConfig):
    base_config: FilePath
    output: OutputFiles = OutputFiles()
    num_trials: int = 50
    options: SearchOptions = SearchOptions()
    experiment_name: str = None
    load_study: FilePath = None
    seed: int = 42

    verbose: bool = False
    quiet: bool = False


def search_from_file(filename) -> Study:
    """Load options from a config file and calls the training procedure.

    Arguments:
        filename: of the configuration file.

    Return:
        an object with training information.
    """
    config = load_config(filename)
    return search_from_configuration(config)


def search_from_configuration(configuration_dict) -> Study:
    """Run the entire training pipeline using the configuration options received.

    Arguments:
        configuration_dict: dictionary with options.

    Return: object with training information.
    """
    config = Configuration(**configuration_dict)

    study = run(config)

    return study


def objective(trial, config: Configuration, kiwi_config: dict) -> float:

    main_metric = 'val_' + '+'.join(kiwi_config['trainer']['main_metric'])

    num_train_lines = sum(
        1 for _ in open(kiwi_config['data']['train']['input']['source'])
    )
    batch_size = kiwi_config['system']['batch_size']
    batches_per_epochs = num_train_lines // batch_size

    kiwi_config['system']['optimizer']['training_steps'] = (
        num_train_lines / batch_size
    ) * kiwi_config['trainer']['epochs']

    # Default searches
    learning_rate = trial.suggest_loguniform(
        'learning_rate',
        config.options.learning_rate.lower,
        config.options.learning_rate.upper,
    )
    dropout = trial.suggest_uniform(
        'dropout', config.options.dropout.lower, config.options.dropout.upper
    )
    warmup_steps = trial.suggest_uniform(
        'warmup_steps',
        config.options.warmup_steps.lower,
        config.options.warmup_steps.upper,
    )
    freeze_epochs = trial.suggest_uniform(
        'freeze_epochs',
        config.options.freeze_epochs.lower,
        config.options.freeze_epochs.upper,
    )

    kiwi_config['system']['optimizer']['learning_rate'] = learning_rate
    kiwi_config['system']['model']['decoder']['dropout'] = dropout
    kiwi_config['system']['model']['outputs']['dropout'] = dropout
    kiwi_config['system']['optimizer']['warmup_steps'] = warmup_steps
    kiwi_config['system']['model']['encoder']['freeze_for_number_of_steps'] = int(
        batches_per_epochs * freeze_epochs
    )

    # Iterable printables
    names = ['learning_rate', 'dropout', 'warmup_steps', 'freeze_epochs']
    values = [learning_rate, dropout, warmup_steps, freeze_epochs]

    if config.options.hidden_size is not None:
        hidden_size = trial.suggest_categorical(
            'hidden_size', config.options.hidden_size
        )
        kiwi_config['system']['model']['encoder']['hidden_size'] = hidden_size
        kiwi_config['system']['model']['decoder']['hidden_size'] = hidden_size
        names.append('hidden_size')
        values.append(hidden_size)

    if config.options.bottleneck_size is not None:
        bottleneck_size = trial.suggest_categorical(
            'bottleneck_size', config.options.bottleneck_size
        )
        kiwi_config['system']['model']['decoder']['bottleneck_size'] = bottleneck_size
        names.append('bottleneck_size')
        values.append(bottleneck_size)

    if config.options.search_mlp:
        use_mlp = trial.suggest_categorical('mlp', [True, False])
        kiwi_config['system']['model']['encoder']['use_mlp'] = use_mlp
        names.append('use_mlp')
        values.append(use_mlp)

    # Search word_level and sentence_level and their combinations
    if config.options.search_hter:
        assert kiwi_config['data']['train']['output']['sentence_scores'] is not None
        assert kiwi_config['data']['valid']['output']['sentence_scores'] is not None
        hter = trial.suggest_categorical('hter', [True, False])
        kiwi_config['system']['model']['outputs']['sentence_level']['hter'] = hter
        names.append('hter')
        values.append(hter)

    if config.options.search_word_level:
        assert kiwi_config['data']['train']['output']['target_tags'] is not None
        assert kiwi_config['data']['valid']['output']['target_tags'] is not None
        word_level = trial.suggest_categorical('word_level', [True, False])
        kiwi_config['system']['model']['outputs']['word_level']['target'] = word_level
        kiwi_config['system']['model']['outputs']['word_level']['gaps'] = word_level
        names.append('word_level')
        values.append(word_level)

    if config.options.search_hter and config.options.search_word_level:
        if hter and word_level and config.options.search_sentence_loss_weight:
            # Also search for the sentence weight
            sentence_loss_weight = trial.suggest_uniform(
                'sentence_loss_weight',
                config.options.sentence_loss_weight.lower,
                config.options.sentence_loss_weight.upper,
            )
            kiwi_config['system']['model']['outputs'][
                'sentence_loss_weight'
            ] = sentence_loss_weight
            names.append('sentence_loss_weight')
            values.append(sentence_loss_weight)

    specified_word_level = kiwi_config['system']['model']['outputs']['word_level'][
        'target'
    ]
    if specified_word_level and config.options.search_hter:
        if hter and config.options.search_sentence_loss_weight:
            # Also search for the sentence weight
            sentence_loss_weight = trial.suggest_uniform(
                'sentence_loss_weight',
                config.options.sentence_loss_weight.lower,
                config.options.sentence_loss_weight.upper,
            )
            kiwi_config['system']['model']['outputs'][
                'sentence_loss_weight'
            ] = sentence_loss_weight
            names.append('sentence_loss_weight')
            values.append(sentence_loss_weight)

    specified_sentence_level = kiwi_config["system"]["model"]["outputs"][
        "sentence_level"
    ]["hter"]
    if specified_sentence_level and config.options.search_sentence_loss_weight:
        # Also search for the sentence weight
        sentence_loss_weight = trial.suggest_uniform(
            'sentence_loss_weight',
            config.options.sentence_loss_weight.lower,
            config.options.sentence_loss_weight.upper,
        )
        kiwi_config["system"]["model"]["outputs"][
            "sentence_loss_weight"
        ] = sentence_loss_weight
        names.append("sentence_loss_weight")
        values.append(sentence_loss_weight)

    if kiwi_config['quiet'] is None:
        kiwi_config['quiet'] = False
    if kiwi_config['verbose'] is None:
        kiwi_config['verbose'] = False
    trainconfig = train.Configuration(**kiwi_config)

    # ptl_callback = PyTorchLightningPruningCallback(trial, monitor='val_PEARSON')
    # train_info = train.run(trainconfig, ptl_callback=ptl_callback)

    logger.info(f"############# STARTING TRIAL {trial.number} ####################### ")
    logger.info(f"PARAMETERS: {dict(zip(names, values))}")
    for name, value in zip(names, values):
        logger.info(f'{name}: {value}')

    try:
        train_info = train.run(trainconfig)
    except RuntimeError as e:
        logger.info(f'ERROR OCCURED; SKIPPING TRIAL: {e}')
        return -1

    logger.info(f"############# TRIAL {trial.number} FINISHED ####################### ")
    result = train_info.best_metrics.get(main_metric, -1)
    if result == -1:
        logger.error(
            'Trial did not converge:\n'
            f'    `train_info.best_metrics={train_info.best_metrics}`\n'
            f'Setting result to -1'
        )

    logger.info(f"RESULTS: {result} {main_metric}")
    logger.info(f"PARAMETERS: {dict(zip(names, values))}")
    for name, value in zip(names, values):
        logger.info(f'{name}: {value}')

    return result


def setup_run(directory: Path, seed: int, debug=False, quiet=False) -> Path:
    if not directory.exists():
        # Initialize a new folder with name '0'
        output_dir = directory / '0'
        output_dir.mkdir(parents=True)
    else:
        # Initialize a new folder incrementing folder count
        output_dir = directory / str(len(list(directory.glob('*'))))
        if output_dir.exists():
            backup_directory = output_dir.with_name(
                f'{output_dir.name}_{datetime.now().isoformat()}'
            )
            logger.warning(
                f'Folder {output_dir} already exists; moving it to {backup_directory}'
            )
        output_dir.mkdir(parents=True)

    logger.info(f'Initializing new search folder at: {output_dir}')

    configure_logging(output_dir=output_dir, verbose=debug, quiet=quiet)
    configure_seed(seed)

    return output_dir


def run(config: Configuration) -> Study:

    output_dir = setup_run(config.output.directory, config.seed)

    logger.info(f'Saving all Optuna results to: {output_dir}')

    # FIXME: this is not neat (but it does work)
    hydra._internal.hydra.GlobalHydra().clear()
    kiwi_config = kiwi.cli.arguments_to_configuration(
        {'CONFIG_FILE': config.base_config}
    )

    if not kiwi_config['run'].get('use_mlflow'):
        logger.warning(
            'You are not using MLflow logging but it is highly advised to do so; '
            'set `run.use_mlflow=true` in the `base_config` file to enable MLflow'
        )

    kiwi_config['trainer']['checkpoint'][
        'early_stop_patience'
    ] = config.options.patience
    kiwi_config['trainer']['checkpoint'][
        'validation_steps'
    ] = config.options.validation_steps

    # Initialize or load a study
    if config.load_study:
        logger.info(f'Loading study to resume from {config.load_study}')
        study = joblib.load(config.load_study)
    else:
        logger.info('Initializing study')
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(
            direction='maximize', pruner=pruner, sampler=TPESampler(seed=config.seed)
        )

    # Optimize the study
    try:
        study.optimize(
            partial(objective, config=config, kiwi_config=kiwi_config),
            n_trials=config.num_trials,
        )
    except Exception as e:
        logger.error(
            f'Error occured during search: {e}; '
            f'current best params are {study.best_params}'
        )
    except KeyboardInterrupt:
        logger.info('Early stopping search caused by ctrl-C')

    logger.info(f"Saving study to {output_dir / 'study.pkl'}")
    joblib.dump(study, output_dir / 'study.pkl')

    try:
        logger.info("Number of finished trials: {}".format(len(study.trials)))
        logger.info("Best trial:")
        trial = study.best_trial
        logger.info("  Value: {}".format(trial.value))
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info("    {}: {}".format(key, value))
    except Exception as e:
        logger.error(f'Logging at end of search failed: {e}')

    logger.info(f'Saving Optuna plots for this search to {output_dir}')
    save_config_to_file(config, output_dir / 'search_config.yaml')
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(output_dir / 'optimization_history.html'))
    except Exception as e:
        logger.error(f'Failed to create plot: {e}')

    try:
        fig = optuna.visualization.plot_parallel_coordinate(
            study, params=list(trial.params.keys())
        )
        fig.write_html(str(output_dir / 'parallel_coordinate.html'))
    except Exception as e:
        logger.error(f'Failed to create plot: {e}')

    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(output_dir / 'param_importances.html'))
    except Exception as e:
        logger.error(f'Failed to create plot: {e}')

    return study
