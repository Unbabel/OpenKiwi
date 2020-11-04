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
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import pydantic
from pydantic import FilePath, validator
from typing_extensions import Literal

from kiwi import constants as const
from kiwi.lib import train
from kiwi.lib.utils import (
    configure_logging,
    configure_seed,
    file_to_configuration,
    save_config_to_file,
    sort_by_second_value,
)
from kiwi.utils.io import BaseConfig

logger = logging.getLogger(__name__)


class RangeConfig(BaseConfig):
    """Specify a continuous interval, or a discrete range when step is set."""

    lower: float
    """The lower bound of the search range."""

    upper: float
    """The upper bound of the search range."""

    step: Optional[float]
    """Specify a step size to create a discrete range of search values."""


class ClassWeightsConfig(BaseConfig):
    """Specify the range to search in for the tag loss weights."""

    target_tags: Union[None, List[float], RangeConfig] = RangeConfig(lower=1, upper=5)
    """Loss weight for the target tags."""

    gap_tags: Union[None, List[float], RangeConfig] = RangeConfig(lower=1, upper=10)
    """Loss weight for the gap tags."""

    source_tags: Union[None, List[float], RangeConfig] = None
    """Loss weight for the source tags."""


class SearchOptions(BaseConfig):
    patience: int = 10
    """Number of training validations without improvement to wait
    before stopping training."""

    validation_steps: float = 0.2
    """Rely on the Kiwi training options to early stop bad models."""

    search_mlp: bool = False
    """To use or not to use an MLP after the encoder."""

    search_word_level: bool = False
    """Try with and without word level output. Useful to figure
    out if word level prediction is helping HTER regression performance."""

    search_hter: bool = False
    """Try with and without sentence level output. Useful to figure
    out if HTER regression is helping word level performance."""

    learning_rate: Union[None, List[float], RangeConfig] = RangeConfig(
        lower=5e-7, upper=5e-5
    )
    """Search the learning rate value."""

    dropout: Union[None, List[float], RangeConfig] = RangeConfig(lower=0.0, upper=0.3)
    """Search the dropout rate used in the decoder."""

    warmup_steps: Union[None, List[float], RangeConfig] = RangeConfig(
        lower=0.05, upper=0.4
    )
    """Search the number of steps to warm up the learning rate."""

    freeze_epochs: Union[None, List[float], RangeConfig] = RangeConfig(lower=0, upper=5)
    """Search the number of epochs to freeze the encoder."""

    class_weights: Union[None, ClassWeightsConfig] = ClassWeightsConfig()
    """Search the word-level tag loss weights."""

    sentence_loss_weight: Union[None, List[float], RangeConfig] = None
    """Search the weight to scale the sentence loss objective with."""

    hidden_size: Union[None, List[int], RangeConfig] = None
    """Search the hidden size of the MLP decoder."""

    bottleneck_size: Union[None, List[int], RangeConfig] = None
    """Search the size of the hidden layer in the decoder bottleneck."""

    search_method: Literal['random', 'tpe', 'multivariate_tpe'] = 'multivariate_tpe'
    """Use random search or the (multivariate) Tree-structured Parzen Estimator,
    or shorthand: TPE. See ``optuna.samplers`` for more details about these methods."""

    @validator('search_hter', pre=True, always=True)
    def check_consistency(cls, v, values):
        if v and values['search_word_level']:
            raise ValueError(
                'Cannot search both word level and sentence level '
                '(``options.search_hter=true`` and ``options.search_word_level=true``) '
                'because there will be no metric that covers both single objectives, '
                'and can lead to no training when neither of the output is selected; '
                'disable one or the other.'
            )
        else:
            return v


class Configuration(BaseConfig):
    base_config: Union[FilePath, train.Configuration]
    """Kiwi train configuration used as a base to configure the search models.
    Can be a path or a yaml configuration properly indented under this argument."""

    directory: Path = Path('optunaruns')
    """Output directory."""

    seed: int = 42
    """Make the search reproducible."""

    search_name: str = None
    """The name used by the Optuna MLflow integration.
    If None, Optuna will create a unique hashed name."""

    num_trials: int = 50
    """The number of search trials to run."""

    num_models_to_keep: int = 5
    """The number of model checkpoints that are kept after finishing search.
    The best checkpoints are kept, the others removed to free up space.
    Keep all model checkpoints by setting this to -1."""

    options: SearchOptions = SearchOptions()
    """Configure the search method and parameter ranges."""

    load_study: FilePath = None
    """Continue from a previous saved study, i.e. from a ``study.pkl`` file."""

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
                    raise pydantic.ValidationError(
                        f'{e}. Configuration field `defaults` from the training '
                        'config is not supported in the search config. Either specify '
                        'the data fields in the regular way, or put the config in '
                        'a separate file and point to the path.'
                    )
        else:
            return Path(v)


def search_from_file(filename: Path):
    """Load options from a config file and calls the training procedure.

    Arguments:
        filename: of the configuration file.

    Returns:
        an object with training information.
    """
    config = file_to_configuration(filename)
    return search_from_configuration(config)


def search_from_configuration(configuration_dict: dict):
    """Run the entire training pipeline using the configuration options received.

    Arguments:
        configuration_dict: dictionary with options.

    Returns:
        object with training information.
    """
    config = Configuration(**configuration_dict)

    study = run(config)

    return study


def get_suggestion(
    trial, param_name: str, config: Union[List, RangeConfig]
) -> Union[bool, float, int]:
    """Let the Optuna trial suggest a parameter value with name ``param_name``
    based on the range configuration.

    Arguments:
        trial: an Optuna trial
        param_name (str): the name of the parameter to suggest a value for
        config (Union[List, RangeConfig]): the parameter search space

    Returns:
        The suggested parameter value.
    """
    if isinstance(config, list):
        return trial.suggest_categorical(param_name, config)
    elif config.step is not None:
        return trial.suggest_discrete_uniform(
            param_name, config.lower, config.upper, config.step
        )
    else:
        return trial.suggest_uniform(param_name, config.lower, config.upper)


def setup_run(directory: Path, seed: int, debug=False, quiet=False) -> Path:
    """Set up the output directory structure for the Optuna search outputs."""
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
                f'{output_dir.name}_backup_{datetime.now().isoformat()}'
            )
            logger.warning(
                f'Folder {output_dir} already exists; moving it to {backup_directory}'
            )
            output_dir.rename(backup_directory)
        output_dir.mkdir(parents=True)

    logger.info(f'Initializing new search folder at: {output_dir}')

    configure_logging(output_dir=output_dir, verbose=debug, quiet=quiet)
    configure_seed(seed)

    return output_dir


class Objective:
    """The objective to be optimized by the Optuna hyperparameter search.

    The call method initializes a Kiwi training config based on Optuna parameter
    suggestions, trains Kiwi, and then returns the output.

    The model paths of the models are saved internally together with the objective
    value obtained for that model. These can be used to prune model checkpoints
    after completion of the search.

    Arguments:
        config (Configuration): the search configuration.
        base_config_dict (dict): the training configuration to serve as base,
            in dictionary form.
    """

    def __init__(self, config: Configuration, base_config_dict: dict):
        self.config = config
        self.base_config_dict = base_config_dict
        self.model_paths = []
        self.train_configs = []

    @property
    def main_metric(self) -> str:
        """The main validation metric as it is formatted by the Kiwi trainer.

        This can be used to access the main metric value after training via
        ``train_info.best_metrics[objective.main_metric]``.
        """
        main_metrics = self.base_config_dict['trainer']['main_metric']
        if not isinstance(main_metrics, list):
            main_metrics = [main_metrics]
        return 'val_' + '+'.join(main_metrics)

    @property
    def num_train_lines(self) -> int:
        """The number of lines in the training data."""
        return sum(
            1 for _ in open(self.base_config_dict['data']['train']['input']['source'])
        )

    @property
    def updates_per_epochs(self) -> int:
        """The number of parameter updates per epochs."""
        return int(
            self.num_train_lines
            / self.base_config_dict['system']['batch_size']['train']
            / self.base_config_dict['trainer']['gradient_accumulation_steps']
        )

    @property
    def best_model_paths(self) -> List[Path]:
        """Return the model paths sorted from high to low by their objective score."""
        return sort_by_second_value(self.model_paths)

    @property
    def best_train_configs(self) -> List[train.Configuration]:
        """Return the train configs sorted from high to low by their objective score."""
        return sort_by_second_value(self.train_configs)

    def prune_models(self, num_models_to_keep) -> None:
        """Keep only the best model checkpoints and remove the rest to free up space."""
        best_model_paths = self.best_model_paths
        for path in best_model_paths[num_models_to_keep:]:
            if Path(path).exists():
                logger.info(
                    f'Removing model checkpoint that is not in the top '
                    f'{num_models_to_keep}: {path}'
                )
                Path(path).unlink()

    def suggest_train_config(self, trial) -> Tuple[train.Configuration, dict]:
        """Use the trial to suggest values to initialize a training configuration.

        Arguments:
            trial: An Optuna trial to make hyperparameter suggestions.

        Return:
            A Kiwi train configuration and a dictionary with the suggested Optuna
            parameter names and values that were set in the train config.
        """
        base_config_dict = deepcopy(self.base_config_dict)

        # Compute the training steps from the training data and set in the base config
        base_config_dict['system']['optimizer']['training_steps'] = (
            self.updates_per_epochs
        ) * base_config_dict['trainer']['epochs']

        # Collect the values to optimize
        search_values = {}

        # Suggest a learning rate
        if self.config.options.learning_rate is not None:
            learning_rate = get_suggestion(
                trial, 'learning_rate', self.config.options.learning_rate
            )
            base_config_dict['system']['optimizer']['learning_rate'] = learning_rate
            search_values['learning_rate'] = learning_rate

        # Suggest a dropout probability
        if self.config.options.dropout is not None:
            dropout = get_suggestion(trial, 'dropout', self.config.options.dropout)
            base_config_dict['system']['model']['outputs']['dropout'] = dropout
            if (
                base_config_dict['system']['model']['encoder'].get('dropout')
                is not None
            ):
                base_config_dict['system']['model']['encoder']['dropout'] = dropout
            if (
                base_config_dict['system']['model']['decoder'].get('dropout')
                is not None
            ):
                base_config_dict['system']['model']['decoder']['dropout'] = dropout
            if base_config_dict['system']['model']['decoder'].get('source') is not None:
                base_config_dict['system']['model']['decoder']['source'][
                    'dropout'
                ] = dropout
            if base_config_dict['system']['model']['decoder'].get('target') is not None:
                base_config_dict['system']['model']['decoder']['target'][
                    'dropout'
                ] = dropout
            search_values['dropout'] = dropout

        # Suggest the number of warmup steps
        if self.config.options.warmup_steps is not None:
            warmup_steps = get_suggestion(
                trial, 'warmup_steps', self.config.options.warmup_steps
            )
            base_config_dict['system']['optimizer']['warmup_steps'] = warmup_steps
            search_values['warmup_steps'] = warmup_steps

        # Suggest the number of freeze epochs
        if self.config.options.freeze_epochs is not None:
            freeze_epochs = get_suggestion(
                trial, 'freeze_epochs', self.config.options.freeze_epochs
            )
            base_config_dict['system']['model']['encoder']['freeze'] = True
            base_config_dict['system']['model']['encoder'][
                'freeze_for_number_of_steps'
            ] = int(self.updates_per_epochs * freeze_epochs)
            search_values['freeze_epochs'] = freeze_epochs

        # Suggest a hidden size
        if self.config.options.hidden_size is not None:
            hidden_size = get_suggestion(
                trial, 'hidden_size', self.config.options.hidden_size
            )
            base_config_dict['system']['model']['encoder']['hidden_size'] = hidden_size
            base_config_dict['system']['model']['decoder']['hidden_size'] = hidden_size
            search_values['hidden_size'] = hidden_size

        # Suggest a bottleneck size
        if self.config.options.bottleneck_size is not None:
            bottleneck_size = get_suggestion(
                trial, 'bottleneck_size', self.config.options.bottleneck_size
            )
            base_config_dict['system']['model']['decoder'][
                'bottleneck_size'
            ] = bottleneck_size
            search_values['bottleneck_size'] = bottleneck_size

        # Suggest whether to use the MLP after the encoder
        if self.config.options.search_mlp:
            use_mlp = trial.suggest_categorical('mlp', [True, False])
            base_config_dict['system']['model']['encoder']['use_mlp'] = use_mlp
            search_values['use_mlp'] = use_mlp

        # Search word_level and sentence_level and their combinations
        # Suggest whether to include the sentence level objective
        if self.config.options.search_hter:
            assert (
                base_config_dict['data']['train']['output']['sentence_scores']
                is not None
            )
            assert (
                base_config_dict['data']['valid']['output']['sentence_scores']
                is not None
            )
            hter = trial.suggest_categorical('hter', [True, False])
            base_config_dict['system']['model']['outputs']['sentence_level'][
                'hter'
            ] = hter
            search_values['hter'] = hter
        else:
            hter = base_config_dict['system']['model']['outputs']['sentence_level'].get(
                'hter'
            )

        # Suggest whether to include the word level objective
        if self.config.options.search_word_level:
            assert (
                base_config_dict['data']['train']['output']['target_tags'] is not None
            )
            assert (
                base_config_dict['data']['valid']['output']['target_tags'] is not None
            )
            word_level = trial.suggest_categorical('word_level', [True, False])
            base_config_dict['system']['model']['outputs']['word_level'][
                'target'
            ] = word_level
            base_config_dict['system']['model']['outputs']['word_level'][
                'gaps'
            ] = word_level
            search_values['word_level'] = word_level

        # We search for `sentence_loss_weight` when there is both a word level
        #   objective and a sentence level objective. Otherwise it does not matter
        #   to weigh the loss of one objective: for the loss it's the ratio between
        #   the two objectives that matters, not the absolute value of one loss
        #   separately.
        word_level_config = base_config_dict['system']['model']['outputs']['word_level']
        sentence_level_config = base_config_dict['system']['model']['outputs'][
            'sentence_level'
        ]
        if (
            word_level_config.get('target')
            and sentence_level_config.get('hter')
            and self.config.options.sentence_loss_weight
            and hter  # The Optuna suggestion that switches hter on or off
        ):
            sentence_loss_weight = get_suggestion(
                trial, 'sentence_loss_weight', self.config.options.sentence_loss_weight
            )
            base_config_dict['system']['model']['outputs'][
                'sentence_loss_weight'
            ] = sentence_loss_weight
            search_values['sentence_loss_weight'] = sentence_loss_weight

        if self.config.options.class_weights and word_level_config:
            for tag_side in ['source', 'target', 'gap']:
                tag_weight_range = self.config.options.class_weights.__dict__.get(
                    f'{tag_side}_tags'
                )
                if word_level_config[tag_side] and tag_weight_range:
                    class_weight = get_suggestion(
                        trial, f'class_weight_{tag_side}_tags', tag_weight_range,
                    )
                    base_config_dict['system']['model']['outputs']['word_level'][
                        'class_weights'
                    ][f'{tag_side}_tags'] = {const.BAD: class_weight}
                    search_values[f'class_weight_{tag_side}_tags'] = class_weight

        return train.Configuration(**base_config_dict), search_values

    def __call__(self, trial) -> float:
        """Train Kiwi with the hyperparameter values suggested by the
        trial and return the value of the main metric.

        Arguments:
            trial: An Optuna trial to make hyperparameter suggestions.

        Returns:
            A float with the value obtained by the Kiwi model,
            as measured by the main metric configured for the model.
        """
        train_config, search_values = self.suggest_train_config(trial)

        logger.info(f'############# STARTING TRIAL {trial.number} #############')
        logger.info(f'PARAMETERS: {search_values}')
        for name, value in search_values.items():
            logger.info(f'{name}: {value}')

        try:
            train_info = train.run(train_config)
        except RuntimeError as e:
            logger.info(f'ERROR OCCURED; SKIPPING TRIAL: {e}')
            return -1

        logger.info(f'############# TRIAL {trial.number} FINISHED #############')
        result = train_info.best_metrics[self.main_metric]

        logger.info(f'RESULTS: {result} {self.main_metric}')
        logger.info(f'MODEL: {train_info.best_model_path}')
        logger.info(f'PARAMETERS: {search_values}')
        for name, value in search_values.items():
            logger.info(f'{name}: {value}')

        # Store the training config for later saving
        self.train_configs.append((train_config, result))

        # Store the checkpoint path and prune the unsucessful model checkpoints
        self.model_paths.append((train_info.best_model_path, result))
        if self.config.num_models_to_keep > 0:
            self.prune_models(self.config.num_models_to_keep)

        return result


def run(config: Configuration):
    """Run hyperparameter search according to the search configuration.

    Args:
        config (Configuration): search configuration

    Return:
        an optuna study summarizing the search results
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            f'{e}. Install the search dependencies with\n'
            'pip install -U openkiwi[search]\n\t'
            'or with\n\t'
            'poetry install -E search\n'
            'when setting up for local development'
        )

    output_dir = setup_run(config.directory, config.seed)
    logger.info(f'Saving all Optuna results to: {output_dir}')
    save_config_to_file(config, output_dir / 'search_config.yaml')

    if isinstance(config.base_config, Path):
        base_dict = file_to_configuration(config.base_config)
        # These arguments are not in the train configuration because they are
        #   added by `kiwi.cli.arguments_to_configuration` (and we remove them
        #   so Pydantic won't throw an error.)
        del base_dict['verbose'], base_dict['quiet']
        base_config = train.Configuration(**base_dict)
    else:
        base_config = config.base_config
    base_config_dict = base_config.dict()

    # Perform some checks of the training config
    use_mlflow = True if base_config_dict['run'].get('use_mlflow') else False
    if not use_mlflow:
        logger.warning(
            'Using MLflow is recommend; set `run.use_mlflow=true` in the base config'
        )
    # The main metric should be explicitly set
    if base_config_dict['trainer']['main_metric'] is None:
        raise ValueError(
            'The metric should be explicitly set in `trainer.main_metric` '
            'in the training config (`base_config`).'
        )

    # Use the early stopping logic of the Kiwi trainer
    base_config_dict['trainer']['checkpoint'][
        'early_stop_patience'
    ] = config.options.patience
    base_config_dict['trainer']['checkpoint'][
        'validation_steps'
    ] = config.options.validation_steps

    # Load or initialize a study
    if config.load_study:
        logger.info(f'Loading study to resume from: {config.load_study}')
        study = joblib.load(config.load_study)
    else:
        if config.options.search_method == 'random':
            logger.info('Exploring parameters with random sampler')
            sampler = optuna.samplers.RandomSampler(seed=config.seed)
        else:
            multivariate = (config.options.search_method == 'multivariate_tpe',)
            logger.info(
                'Exploring parameters with '
                f'{"multivariate " if multivariate else ""}TPE sampler'
            )
            sampler = optuna.samplers.TPESampler(
                seed=config.seed, multivariate=multivariate,
            )
        logger.info('Initializing study...')
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(
            study_name=config.search_name,
            direction='maximize',
            pruner=pruner,
            sampler=sampler,
        )

    # Optimize the study
    optimize_kwargs = {}
    if use_mlflow:
        # Use MLflow integration
        optimize_kwargs['callbacks'] = [optuna.integration.MLflowCallback()]
        # NOTE: we tried the integration with the PTL trainer callback,
        #   but the early stopping of the training was simpler
        #   and more effective in the end
    try:
        logger.info('Optimizing study...')
        objective = Objective(config=config, base_config_dict=base_config_dict)
        study.optimize(
            objective, n_trials=config.num_trials, **optimize_kwargs,
        )
    except KeyboardInterrupt:
        logger.info('Early stopping search caused by ctrl-C')
    except Exception as e:
        logger.error(
            f'Error occured during search: {e}; '
            f'current best params are: {study.best_params}'
        )

    # Save the study objective for possible later continuation
    logger.info(f"Saving study to: {output_dir / 'study.pkl'}")
    joblib.dump(study, output_dir / 'study.pkl')

    # Log the found values of the best search trial
    logger.info(f'Number of finished trials: {len(study.trials)}')
    logger.info('Best trial:')
    logger.info(f'  Value: {study.best_trial.value}')
    logger.info('  Params:')
    for key, value in study.best_trial.params.items():
        logger.info(f'    {key}: {value}')

    # Save the training configs for the best models to file
    configs_dir = output_dir / 'best_configs'
    logger.info(
        f'Saving best {config.num_models_to_keep} train configs to: {configs_dir}'
    )
    configs_dir.mkdir()
    for i, train_config in enumerate(
        objective.best_train_configs[: config.num_models_to_keep]
    ):
        save_config_to_file(train_config, configs_dir / f'train_{i}.yaml')

    # Create and save Optuna search plots
    logger.info(f'Saving Optuna plots for this search to: {output_dir}')

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(str(output_dir / 'optimization_history.html'))

    fig = optuna.visualization.plot_parallel_coordinate(
        study, params=list(study.best_trial.params.keys())
    )
    fig.write_html(str(output_dir / 'parallel_coordinate.html'))

    if len(study.trials) > 2:
        try:
            # This fits a random forest regression model using sklearn to predict the
            #   importance of each parameter
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(str(output_dir / 'param_importances.html'))
        except RuntimeError as e:
            logger.info(
                f'Error training the regression model '
                f'to compute the parameter importances: {e}'
            )

    return study
