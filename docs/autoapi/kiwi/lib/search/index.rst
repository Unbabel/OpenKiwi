:mod:`kiwi.lib.search`
======================

.. py:module:: kiwi.lib.search


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.lib.search.RangeConfig
   kiwi.lib.search.ClassWeightsConfig
   kiwi.lib.search.SearchOptions
   kiwi.lib.search.Configuration
   kiwi.lib.search.Objective



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.lib.search.search_from_file
   kiwi.lib.search.search_from_configuration
   kiwi.lib.search.get_suggestion
   kiwi.lib.search.setup_run
   kiwi.lib.search.run


.. data:: logger
   

   

.. py:class:: RangeConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Specify a continuous interval, or a discrete range when step is set.

   .. attribute:: lower
      :annotation: :float

      The lower bound of the search range.


   .. attribute:: upper
      :annotation: :float

      The upper bound of the search range.


   .. attribute:: step
      :annotation: :Optional[float]

      Specify a step size to create a discrete range of search values.



.. py:class:: ClassWeightsConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Specify the range to search in for the tag loss weights.

   .. attribute:: target_tags
      :annotation: :Union[None, List[float], RangeConfig]

      Loss weight for the target tags.


   .. attribute:: gap_tags
      :annotation: :Union[None, List[float], RangeConfig]

      Loss weight for the gap tags.


   .. attribute:: source_tags
      :annotation: :Union[None, List[float], RangeConfig]

      Loss weight for the source tags.



.. py:class:: SearchOptions

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: patience
      :annotation: :int = 10

      Number of training validations without improvement to wait
      before stopping training.


   .. attribute:: validation_steps
      :annotation: :float = 0.2

      Rely on the Kiwi training options to early stop bad models.


   .. attribute:: search_mlp
      :annotation: :bool = False

      To use or not to use an MLP after the encoder.


   .. attribute:: search_word_level
      :annotation: :bool = False

      Try with and without word level output. Useful to figure
      out if word level prediction is helping HTER regression performance.


   .. attribute:: search_hter
      :annotation: :bool = False

      Try with and without sentence level output. Useful to figure
      out if HTER regression is helping word level performance.


   .. attribute:: learning_rate
      :annotation: :Union[None, List[float], RangeConfig]

      Search the learning rate value.


   .. attribute:: dropout
      :annotation: :Union[None, List[float], RangeConfig]

      Search the dropout rate used in the decoder.


   .. attribute:: warmup_steps
      :annotation: :Union[None, List[float], RangeConfig]

      Search the number of steps to warm up the learning rate.


   .. attribute:: freeze_epochs
      :annotation: :Union[None, List[float], RangeConfig]

      Search the number of epochs to freeze the encoder.


   .. attribute:: class_weights
      :annotation: :Union[None, ClassWeightsConfig]

      Search the word-level tag loss weights.


   .. attribute:: sentence_loss_weight
      :annotation: :Union[None, List[float], RangeConfig]

      Search the weight to scale the sentence loss objective with.


   .. attribute:: hidden_size
      :annotation: :Union[None, List[int], RangeConfig]

      Search the hidden size of the MLP decoder.


   .. attribute:: bottleneck_size
      :annotation: :Union[None, List[int], RangeConfig]

      Search the size of the hidden layer in the decoder bottleneck.


   .. attribute:: search_method
      :annotation: :Literal['random', 'tpe', 'multivariate_tpe'] = multivariate_tpe

      Use random search or the (multivariate) Tree-structured Parzen Estimator,
      or shorthand: TPE. See ``optuna.samplers`` for more details about these methods.


   .. method:: check_consistency(cls, v, values)



.. py:class:: Configuration

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: base_config
      :annotation: :Union[FilePath, train.Configuration]

      Kiwi train configuration used as a base to configure the search models.
      Can be a path or a yaml configuration properly indented under this argument.


   .. attribute:: directory
      :annotation: :Path

      Output directory.


   .. attribute:: seed
      :annotation: :int = 42

      Make the search reproducible.


   .. attribute:: search_name
      :annotation: :str

      The name used by the Optuna MLflow integration.
      If None, Optuna will create a unique hashed name.


   .. attribute:: num_trials
      :annotation: :int = 50

      The number of search trials to run.


   .. attribute:: num_models_to_keep
      :annotation: :int = 5

      The number of model checkpoints that are kept after finishing search.
      The best checkpoints are kept, the others removed to free up space.
      Keep all model checkpoints by setting this to -1.


   .. attribute:: options
      :annotation: :SearchOptions

      Configure the search method and parameter ranges.


   .. attribute:: load_study
      :annotation: :FilePath

      Continue from a previous saved study, i.e. from a ``study.pkl`` file.


   .. attribute:: verbose
      :annotation: :bool = False

      

   .. attribute:: quiet
      :annotation: :bool = False

      

   .. method:: parse_base_config(cls, v)



.. function:: search_from_file(filename: Path)

   Load options from a config file and calls the training procedure.

   :param filename: of the configuration file.

   :returns: an object with training information.


.. function:: search_from_configuration(configuration_dict: dict)

   Run the entire training pipeline using the configuration options received.

   :param configuration_dict: dictionary with options.

   :returns: object with training information.


.. function:: get_suggestion(trial, param_name: str, config: Union[List, RangeConfig]) -> Union[bool, float, int]

   Let the Optuna trial suggest a parameter value with name ``param_name``
   based on the range configuration.

   :param trial: an Optuna trial
   :param param_name: the name of the parameter to suggest a value for
   :type param_name: str
   :param config: the parameter search space
   :type config: Union[List, RangeConfig]

   :returns: The suggested parameter value.


.. function:: setup_run(directory: Path, seed: int, debug=False, quiet=False) -> Path

   Set up the output directory structure for the Optuna search outputs.


.. py:class:: Objective(config: Configuration, base_config_dict: dict)

   The objective to be optimized by the Optuna hyperparameter search.

   The call method initializes a Kiwi training config based on Optuna parameter
   suggestions, trains Kiwi, and then returns the output.

   The model paths of the models are saved internally together with the objective
   value obtained for that model. These can be used to prune model checkpoints
   after completion of the search.

   :param config: the search configuration.
   :type config: Configuration
   :param base_config_dict: the training configuration to serve as base,
                            in dictionary form.
   :type base_config_dict: dict

   .. method:: main_metric(self) -> str
      :property:

      The main validation metric as it is formatted by the Kiwi trainer.

      This can be used to access the main metric value after training via
      ``train_info.best_metrics[objective.main_metric]``.


   .. method:: num_train_lines(self) -> int
      :property:

      The number of lines in the training data.


   .. method:: updates_per_epochs(self) -> int
      :property:

      The number of parameter updates per epochs.


   .. method:: best_model_paths(self) -> List[Path]
      :property:

      Return the model paths sorted from high to low by their objective score.


   .. method:: suggest_train_config(self, trial) -> Tuple[train.Configuration, dict]

      Use the trial to suggest values to initialize a training configuration.

      :param trial: An Optuna trial to make hyperparameter suggestions.

      :returns: A Kiwi train configuration and a dictionary with the suggested Optuna
                parameter names and values that were set in the train config.


   .. method:: __call__(self, trial) -> float

      Train Kiwi with the hyperparameter values suggested by the
      trial and return the value of the main metric.

      :param trial: An Optuna trial to make hyperparameter suggestions.

      :returns: A float with the value obtained by the Kiwi model,
                as measured by the main metric configured for the model.



.. function:: run(config: Configuration)


