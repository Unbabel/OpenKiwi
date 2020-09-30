:mod:`kiwi.lib.train`
=====================

.. py:module:: kiwi.lib.train


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.lib.train.TrainRunInfo
   kiwi.lib.train.RunConfig
   kiwi.lib.train.CheckpointsConfig
   kiwi.lib.train.GPUConfig
   kiwi.lib.train.TrainerConfig
   kiwi.lib.train.Configuration



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.lib.train.train_from_file
   kiwi.lib.train.train_from_configuration
   kiwi.lib.train.setup_run
   kiwi.lib.train.run


.. data:: logger
   

   

.. py:class:: TrainRunInfo

   Encapsulate relevant information on training runs.

   .. attribute:: model
      :annotation: :QESystem

      The last model when training finished.


   .. attribute:: best_metrics
      :annotation: :Dict[str, Any]

      Mapping of metrics of the best model.


   .. attribute:: best_model_path
      :annotation: :Optional[Path]

      Path of the best model, if it was saved to disk.



.. py:class:: RunConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Options for each run.

   .. attribute:: seed
      :annotation: :int = 42

      Random seed


   .. attribute:: experiment_name
      :annotation: :str = default

      If using MLflow, it will log this run under this experiment name, which appears
      as a separate section in the UI. It will also be used in some messages and files.


   .. attribute:: output_dir
      :annotation: :Path

      Output several files for this run under this directory.
      If not specified, a directory under "./runs/" is created or reused based on the
      ``run_id``. Files might also be sent to MLflow depending on the
      ``mlflow_always_log_artifacts`` option.


   .. attribute:: run_id
      :annotation: :str

      If specified, MLflow/Default Logger will log metrics and params
      under this ID. If it exists, the run status will change to running.
      This ID is also used for creating this run's output directory if
      ``output_dir`` is not specified (Run ID must be a 32-character hex string).


   .. attribute:: use_mlflow
      :annotation: :bool = False

      Whether to use MLflow for tracking this run. If not installed, a message
      is shown


   .. attribute:: mlflow_tracking_uri
      :annotation: :str = mlruns/

      If using MLflow, logs model parameters, training metrics, and
      artifacts (files) to this MLflow server. Uses the localhost by
      default.


   .. attribute:: mlflow_always_log_artifacts
      :annotation: :bool = False

      If using MLFlow, always log (send) artifacts (files) to MLflow
      artifacts URI. By default (false), artifacts are only logged if
      MLflow is a remote server (as specified by --mlflow-tracking-uri
      option).All generated files are always saved in --output-dir, so it
      might be considered redundant to copy them to a local MLflow
      server. If this is not the case, set this option to true.



.. py:class:: CheckpointsConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: validation_steps
      :annotation: :Union[confloat(gt=0.0, le=1.0), PositiveInt] = 1.0

      How often within one training epoch to check the validation set.
      If float, % of training epoch. If int, check every n batches.


   .. attribute:: save_top_k
      :annotation: :int = 1

      Save and keep only ``k`` best models according to main metric;
      -1 will keep all; 0 will never save a model.


   .. attribute:: early_stop_patience
      :annotation: :conint(ge=0) = 0

      Stop training if evaluation metrics do not improve after X validations;
      0 disables this.



.. py:class:: GPUConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: gpus
      :annotation: :Union[int, List[int]] = 0

      Use the number of GPUs specified if int, where 0 is no GPU. -1 is all GPUs.
      Alternatively, if a list, uses the GPU-ids specified (e.g., [0, 2]).


   .. attribute:: precision
      :annotation: :Literal[16, 32] = 32

      The floating point precision to be used while training the model. Available
      options are 32 or 16 bits.


   .. attribute:: amp_level
      :annotation: :Literal['O0', 'O1', 'O2', 'O3'] = O0

      The automatic-mixed-precision level to use. O0 is FP32 training. 01 is mixed
      precision training as popularized by NVIDIA Apex. O2 casts the model weights to FP16
       but keeps certain master weights and batch norm in FP32 without patching Torch
      functions. 03 is full FP16 training.


   .. method:: setup_gpu_ids(cls, v)

      If asking to use CPU, let it be, outputting a warning if GPUs are available.
      If asking to use any GPU but none are available, fall back to CPU and warn user.


   .. method:: setup_amp_level(cls, v, values)

      If precision is set to 16, amp_level needs to be greater than O0.
      Following the same logic, if amp_level is set to greater than O0, precision
      needs to be set to 16.



.. py:class:: TrainerConfig

   Bases: :class:`kiwi.lib.train.GPUConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: resume
      :annotation: :bool = False

      Resume training a previous run.
      The `run.run_id` (and possibly `run.experiment_name`) option must be specified.
      Files are then searched under the "runs" directory. If not found, they are
      downloaded from the MLflow server (check the `mlflow_tracking_uri` option).


   .. attribute:: epochs
      :annotation: :int = 50

      Number of epochs for training.


   .. attribute:: gradient_accumulation_steps
      :annotation: :int = 1

      Accumulate gradients for the given number of steps (batches) before
      back-propagating.


   .. attribute:: gradient_max_norm
      :annotation: :float = 0.0

      Clip gradients with norm above this value; by default (0.0), do not clip.


   .. attribute:: main_metric
      :annotation: :Union[str, List[str]]

      Choose Primary Metric for this run.


   .. attribute:: log_interval
      :annotation: :int = 100

      Log every k batches.


   .. attribute:: log_save_interval
      :annotation: :int = 100

      Save accumulated log every k batches (does not seem to
      matter to MLflow logging).


   .. attribute:: checkpoint
      :annotation: :CheckpointsConfig

      

   .. attribute:: deterministic
      :annotation: :bool = True

      If true enables cudnn.deterministic. Might make training slower, but ensures
      reproducibility.



.. py:class:: Configuration

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: run
      :annotation: :RunConfig

      Options specific to each run


   .. attribute:: trainer
      :annotation: :TrainerConfig

      

   .. attribute:: data
      :annotation: :WMTQEDataset.Config

      

   .. attribute:: system
      :annotation: :QESystem.Config

      

   .. attribute:: debug
      :annotation: :bool = False

      Run training in `fast_dev` mode; only one batch is used for training and
      validation. This is useful to test out new models.


   .. attribute:: verbose
      :annotation: :bool = False

      

   .. attribute:: quiet
      :annotation: :bool = False

      


.. function:: train_from_file(filename) -> TrainRunInfo

   Load options from a config file and calls the training procedure.

   :param filename: of the configuration file.

   :returns: an object with training information.


.. function:: train_from_configuration(configuration_dict) -> TrainRunInfo

   Run the entire training pipeline using the configuration options received.

   :param configuration_dict: dictionary with options.

   Return: object with training information.


.. function:: setup_run(config: RunConfig, debug=False, quiet=False, anchor_dir: Path = None) -> Tuple[Path, Optional[MLFlowTrackingLogger]]

   Prepare for running the training pipeline.

   This includes setting up the output directory, random seeds, and loggers.

   :param config: configuration options.
   :param quiet: whether to suppress info log messages.
   :param debug: whether to additionally log debug messages
                 (:param:`quiet` has precedence)
   :param anchor_dir: directory to use as root for paths.

   :returns: a tuple with the resolved path to the output directory and the experiment
             logger (``None`` if not configured).


.. function:: run(config: Configuration, system_type: Union[Type[TLMSystem], Type[QESystem]] = QESystem) -> TrainRunInfo

   Instantiate the system according to the configuration and train it.

   Load or create a trainer for doing it.

   :param config: generic training options.
   :param system_type: class of system being used.

   :returns: an object with training information.


