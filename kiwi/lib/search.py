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
from typing import List, Union

import hydra
import joblib
import pydantic
from pydantic import FilePath, validator
from typing_extensions import Literal

import kiwi.cli
from kiwi import constants as const
from kiwi.lib import train
from kiwi.lib.utils import (
    configure_logging,
    configure_seed,
    load_config,
    save_config_to_file,
)
from kiwi.utils.io import BaseConfig

logger = logging.getLogger(__name__)


class RangeConfig(BaseConfig):
    lower: float
    upper: float


class ClassWeightsConfig(BaseConfig):
    """Specify the range to search in for the tag loss weights."""

    target_tags: RangeConfig = RangeConfig(lower=1, upper=5)
    gap_tags: RangeConfig = RangeConfig(lower=1, upper=10)
    source_tags: RangeConfig = None


class SearchOptions(BaseConfig):
    patience: int = 10
    validation_steps: float = 0.2
    """Rely on the Kiwi training options to eartly stop bad models."""

    search_mlp: bool = False
    """To use or not to use an MLP after the encoder."""

    search_word_level: bool = False
    """Try with and without word level output. Useful to figure
    out if word level prediction is helping HTER regression performance."""

    search_hter: bool = False
    """Try with and without sentence level output. Useful to figure
    out if HTER regression is helping word level performance."""

    search_sentence_loss_weight: bool = False
    """Search the weight by which the sentence level loss is scaled.
    If ``search_sentence_loss_weight`` is false then the range specified
    in ``sentence_loss_weight`` will be searched over."""

    learning_rate: Union[None, RangeConfig] = RangeConfig(lower=5e-7, upper=5e-5)
    dropout: Union[None, RangeConfig] = RangeConfig(lower=0.0, upper=0.3)
    warmup_steps: Union[None, RangeConfig] = RangeConfig(lower=0.05, upper=0.4)
    freeze_epochs: Union[None, RangeConfig] = RangeConfig(lower=0, upper=5)
    class_weights: Union[None, ClassWeightsConfig] = ClassWeightsConfig()
    sentence_loss_weight: Union[None, RangeConfig] = None
    """Specify the ranges to search in."""

    hidden_size: List[int] = None
    bottleneck_size: List[int] = None
    """List the integers to search over."""

    search_method: Literal['random', 'tpe', 'multivariate_tpe'] = 'multivariate_tpe'
    """Use random search or the (multivariate) Tree-structured Parzen Estimator,
    or sharthand: TPE. See optuna.samplers for more details about these methods."""


class Configuration(BaseConfig):
    base_config: Union[FilePath, train.Configuration]
    """Kiwi train configuration used as a base to configure the model.
    Pass the path or put the entire training config under this argument."""

    directory: Path = Path('optunaruns')
    """Output directory."""

    seed: int = 42
    search_name: str = None
    num_trials: int = 50
    """General configuration of the search."""

    options: SearchOptions = SearchOptions()
    """Configure the search method and parameter ranges."""

    load_study: FilePath = None
    """Continue from a previous saved study, i.e. from a `study.pkl` file."""

    verbose: bool = False
    quiet: bool = False

    @validator('base_config', pre=True, always=True)
    def parse_base_config(cls, v):
        if isinstance(v, dict):
            try:
                return train.Configuration(**v)
            except pydantic.error_wrappers.ValidationError as e:
                logger.info(str(e))
                if 'defaults' in str(e):
                    logger.error(
                        'Configuration field `defaults` from the training config '
                        'is not supported in the search config; specify the data '
                        'fields in the regular way'
                    )
                exit()
        else:
            return Path(v)


def search_from_file(filename):
    """Load options from a config file and calls the training procedure.

    Arguments:
        filename: of the configuration file.

    Return:
        an object with training information.
    """
    config = load_config(filename)
    return search_from_configuration(config)


def search_from_configuration(configuration_dict):
    """Run the entire training pipeline using the configuration options received.

    Arguments:
        configuration_dict: dictionary with options.

    Return: object with training information.
    """
    config = Configuration(**configuration_dict)

    study = run(config)

    return study


def objective(trial, config: Configuration, base_config: dict) -> float:

    # Format the main metric name
    main_metric = 'val_' + '+'.join(base_config['trainer']['main_metric'])

    # Set the training steps
    num_train_lines = sum(
        1 for _ in open(base_config['data']['train']['input']['source'])
    )
    batch_size = base_config['system']['batch_size']['train']
    updates_per_epochs = num_train_lines // batch_size
    if base_config['trainer'].get('gradient_accumulation_steps'):
        updates_per_epochs //= base_config['trainer']['gradient_accumulation_steps']
    base_config['system']['optimizer']['training_steps'] = (
        updates_per_epochs
    ) * base_config['trainer']['epochs']

    # Collect the values to optimize
    search_values = {}
    if config.options.learning_rate is not None:
        learning_rate = trial.suggest_loguniform(
            'learning_rate',
            config.options.learning_rate.lower,
            config.options.learning_rate.upper,
        )
        base_config['system']['optimizer']['learning_rate'] = learning_rate
        search_values['learning_rate'] = learning_rate

    if config.options.dropout is not None:
        dropout = trial.suggest_uniform(
            'dropout', config.options.dropout.lower, config.options.dropout.upper
        )
        if 'dropout' in base_config['system']['model']['outputs']:
            base_config['system']['model']['outputs']['dropout'] = dropout
        if 'dropout' in base_config['system']['model']['decoder']:
            base_config['system']['model']['decoder']['dropout'] = dropout
        search_values['dropout'] = dropout

    if config.options.warmup_steps is not None:
        warmup_steps = trial.suggest_uniform(
            'warmup_steps',
            config.options.warmup_steps.lower,
            config.options.warmup_steps.upper,
        )
        base_config['system']['optimizer']['warmup_steps'] = warmup_steps
        search_values['warmup_steps'] = warmup_steps

    if config.options.freeze_epochs is not None:
        freeze_epochs = trial.suggest_uniform(
            'freeze_epochs',
            config.options.freeze_epochs.lower,
            config.options.freeze_epochs.upper,
        )
        base_config['system']['model']['encoder']['freeze_for_number_of_steps'] = int(
            updates_per_epochs * freeze_epochs
        )
        search_values['freeze_epochs'] = freeze_epochs

    if config.options.hidden_size is not None:
        hidden_size = trial.suggest_categorical(
            'hidden_size', config.options.hidden_size
        )
        base_config['system']['model']['encoder']['hidden_size'] = hidden_size
        base_config['system']['model']['decoder']['hidden_size'] = hidden_size
        search_values['hidden_size'] = hidden_size

    if config.options.bottleneck_size is not None:
        bottleneck_size = trial.suggest_categorical(
            'bottleneck_size', config.options.bottleneck_size
        )
        base_config['system']['model']['decoder']['bottleneck_size'] = bottleneck_size
        search_values['bottleneck_size'] = bottleneck_size

    if config.options.search_mlp:
        use_mlp = trial.suggest_categorical('mlp', [True, False])
        base_config['system']['model']['encoder']['use_mlp'] = use_mlp
        search_values['use_mlp'] = use_mlp

    # Search word_level and sentence_level and their combinations
    if config.options.search_hter:
        assert base_config['data']['train']['output']['sentence_scores'] is not None
        assert base_config['data']['valid']['output']['sentence_scores'] is not None
        hter = trial.suggest_categorical('hter', [True, False])
        base_config['system']['model']['outputs']['sentence_level']['hter'] = hter
        search_values['hter'] = hter

    if config.options.search_word_level:
        assert base_config['data']['train']['output']['target_tags'] is not None
        assert base_config['data']['valid']['output']['target_tags'] is not None
        word_level = trial.suggest_categorical('word_level', [True, False])
        base_config['system']['model']['outputs']['word_level']['target'] = word_level
        base_config['system']['model']['outputs']['word_level']['gaps'] = word_level
        search_values['word_level'] = word_level

    if config.options.search_hter and config.options.search_word_level:
        if hter and word_level and config.options.search_sentence_loss_weight:
            # Also search for the sentence weight
            sentence_loss_weight = trial.suggest_uniform(
                'sentence_loss_weight',
                config.options.sentence_loss_weight.lower,
                config.options.sentence_loss_weight.upper,
            )
            base_config['system']['model']['outputs'][
                'sentence_loss_weight'
            ] = sentence_loss_weight
            search_values['sentence_loss_weight'] = sentence_loss_weight

    specified_word_level = base_config['system']['model']['outputs']['word_level'][
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
            base_config['system']['model']['outputs'][
                'sentence_loss_weight'
            ] = sentence_loss_weight
            search_values['sentence_loss_weight'] = sentence_loss_weight

    specified_sentence_level = base_config['system']['model']['outputs'][
        'sentence_level'
    ]['hter']
    if specified_sentence_level and config.options.search_sentence_loss_weight:
        # Also search for the sentence weight
        sentence_loss_weight = trial.suggest_uniform(
            'sentence_loss_weight',
            config.options.sentence_loss_weight.lower,
            config.options.sentence_loss_weight.upper,
        )
        base_config['system']['model']['outputs'][
            'sentence_loss_weight'
        ] = sentence_loss_weight
        search_values['sentence_loss_weight'] = sentence_loss_weight

    if config.options.class_weights is not None:
        # if config.options.search_word_level:
        #     if not word_level:
        #         # Word level disabled; nothing to search over
        #         pass
        if config.options.class_weights.target_tags:
            class_weight_target_tags = trial.suggest_uniform(
                'class_weight_target_tags',
                config.options.class_weights.target_tags.lower,
                config.options.class_weights.target_tags.upper,
            )
            base_config['system']['model']['outputs']['word_level']['class_weights'][
                'target_tags'
            ] = {const.BAD: class_weight_target_tags}
            search_values['class_weight_target_tags'] = class_weight_target_tags
        if config.options.class_weights.gap_tags:
            class_weight_gap_tags = trial.suggest_uniform(
                'class_weight_gap_tags',
                config.options.class_weights.gap_tags.lower,
                config.options.class_weights.gap_tags.upper,
            )
            base_config['system']['model']['outputs']['word_level']['class_weights'][
                'gap_tags'
            ] = {const.BAD: class_weight_gap_tags}
            search_values['class_weight_gap_tags'] = class_weight_gap_tags
        if config.options.class_weights.source_tags:
            class_weight_source_tags = trial.suggest_uniform(
                'class_weight_source_tags',
                config.options.class_weights.source_tags.lower,
                config.options.class_weights.source_tags.upper,
            )
            base_config['system']['model']['outputs']['word_level']['class_weights'][
                'source_tags'
            ] = {const.BAD: class_weight_source_tags}
            search_values['class_weight_source_tags'] = class_weight_source_tags

    if base_config['quiet'] is None:
        base_config['quiet'] = False
    if base_config['verbose'] is None:
        base_config['verbose'] = False
    trainconfig = train.Configuration(**base_config)

    # TODO: integrate PTL trainer callback
    # ptl_callback = PyTorchLightningPruningCallback(trial, monitor='val_PEARSON')
    # train_info = train.run(trainconfig, ptl_callback=ptl_callback)

    logger.info(f'############# STARTING TRIAL {trial.number} #############')
    logger.info(f'PARAMETERS: {search_values}')
    for name, value in search_values.items():
        logger.info(f'{name}: {value}')

    try:
        train_info = train.run(trainconfig)
    except RuntimeError as e:
        logger.info(f'ERROR OCCURED; SKIPPING TRIAL: {e}')
        return -1

    logger.info(f'############# TRIAL {trial.number} FINISHED #############')
    result = train_info.best_metrics.get(main_metric, -1)
    if result == -1:
        logger.error(
            'Trial did not converge:\n'
            f'    `train_info.best_metrics={train_info.best_metrics}`\n'
            f'Setting result to -1'
        )

    logger.info(f'RESULTS: {result} {main_metric}')
    logger.info(f'PARAMETERS: {search_values}')
    for name, value in search_values.items():
        logger.info(f'{name}: {value}')

    return result


def setup_run(directory: Path, seed: int, debug=False, quiet=False) -> Path:
    # TODO: replace all of this with the MLFlow integration: optuna.integration.mlflow
    #   In particular: let the output directory be created by MLflow entirely
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


def run(config: Configuration):
    try:
        import optuna
    except ImportError as e:
        logger.error(
            f'ImportError: {e}. Install the search dependencies with '
            '``pip install -U openkiwi[search]``, or with ``poetry install -E search`` '
            'when setting up for local development'
        )
        exit()

    output_dir = setup_run(config.directory, config.seed)

    logger.info(f'Saving all Optuna results to: {output_dir}')

    if isinstance(config.base_config, Path):
        # FIXME: this is not neat (but it does work)
        hydra._internal.hydra.GlobalHydra().clear()
        base_dict = kiwi.cli.arguments_to_configuration(
            {'CONFIG_FILE': config.base_config}
        )
        del base_dict['verbose'], base_dict['quiet']
        base_config = train.Configuration(**base_dict)
    else:
        base_config = config.base_config
    base_config = base_config.dict()

    # Perform some checks of the training config
    if not base_config['run'].get('use_mlflow'):
        logger.info('Setting `run.use_mlflow=true` in the base config')
        base_config['run']['use_mlflow'] = True
    if base_config['trainer']['main_metric'] is None:
        logger.error(
            'The metric should be explicitly set in `trainer.main_metric` '
            'in the training config (`base_config`).'
        )
        exit()
    if not isinstance(base_config['trainer']['main_metric'], list):
        base_config['trainer']['main_metric'] = [base_config['trainer']['main_metric']]

    base_config['trainer']['checkpoint'][
        'early_stop_patience'
    ] = config.options.patience
    base_config['trainer']['checkpoint'][
        'validation_steps'
    ] = config.options.validation_steps

    # Initialize or load a study
    if config.load_study:
        logger.info(f'Loading study to resume from: {config.load_study}')
        study = joblib.load(config.load_study)
    else:
        if config.options.search_method == 'tpe':
            logger.info('Exploring parameters with TPE sampler')
            sampler = optuna.samplers.TPESampler(seed=config.seed)
        elif config.options.search_method == 'multivariate_tpe':
            logger.info('Exploring parameters with multivariate TPE sampler')
            sampler = optuna.samplers.TPESampler(seed=config.seed, multivariate=True)
        else:
            logger.info('Exploring parameters with random sampler')
            sampler = optuna.samplers.RandomSampler(seed=config.seed)

        logger.info('Initializing study...')
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(
            study_name=config.search_name,
            direction='maximize',
            pruner=pruner,
            sampler=sampler,
        )

    # Optimize the study
    # TODO: keep only n best model checkpoints and remove the rest to free up space
    mlflc = optuna.integration.MLflowCallback()
    try:
        logger.info('Optimizing study...')
        study.optimize(
            partial(objective, config=config, base_config=base_config),
            n_trials=config.num_trials,
            callbacks=[mlflc],
        )
    except KeyboardInterrupt:
        logger.info('Early stopping search caused by ctrl-C')
    except Exception as e:
        logger.error(
            f'Error occured during search: {e}; '
            f'current best params are: {study.best_params}'
        )

    logger.info(f"Saving study to: {output_dir / 'study.pkl'}")
    joblib.dump(study, output_dir / 'study.pkl')

    try:
        logger.info('Number of finished trials: {}'.format(len(study.trials)))
        logger.info('Best trial:')
        trial = study.best_trial
        logger.info('  Value: {}'.format(trial.value))
        logger.info('  Params: ')
        for key, value in trial.params.items():
            logger.info('    {}: {}'.format(key, value))
    except Exception as e:
        logger.error(f'Logging at end of search failed: {e}')

    logger.info(f'Saving Optuna plots for this search to: {output_dir}')
    save_config_to_file(config, output_dir / 'search_config.yaml')
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(output_dir / 'optimization_history.html'))
    except Exception as e:
        logger.error(f'Failed to create plot `optimization_history`: {e}')

    try:
        fig = optuna.visualization.plot_parallel_coordinate(
            study, params=list(trial.params.keys())
        )
        fig.write_html(str(output_dir / 'parallel_coordinate.html'))
    except Exception as e:
        logger.error(f'Failed to create plot `parallel_coordinate`: {e}')

    try:
        # This fits a random forest regression model using sklearn to predict the
        #   importance of each parameter
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(output_dir / 'param_importances.html'))
    except Exception as e:
        logger.error(f'Failed to create plot `param_importances`: {e}')

    return study
