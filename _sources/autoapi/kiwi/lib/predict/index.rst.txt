:mod:`kiwi.lib.predict`
=======================

.. py:module:: kiwi.lib.predict


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.lib.predict.RunConfig
   kiwi.lib.predict.Configuration



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.lib.predict.load_system
   kiwi.lib.predict.predict_from_configuration
   kiwi.lib.predict.run
   kiwi.lib.predict.make_predictions
   kiwi.lib.predict.setup_run


.. data:: logger
   

   

.. py:class:: RunConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: seed
      :annotation: :int = 42

      Random seed


   .. attribute:: run_id
      :annotation: :str

      If specified, MLflow/Default Logger will log metrics and params
      under this ID. If it exists, the run status will change to running.
      This ID is also used for creating this run's output directory.
      (Run ID must be a 32-character hex string).


   .. attribute:: output_dir
      :annotation: :Path

      Output several files for this run under this directory.
      If not specified, a directory under "runs" is created or reused based on the
      Run UUID.


   .. attribute:: predict_on_data_partition
      :annotation: :Literal['train', 'valid', 'test'] = test

      Name of the data partition to predict upon. File names are read from the
      corresponding ``data`` configuration field.


   .. method:: check_consistency(cls, v, values)



.. py:class:: Configuration

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: run
      :annotation: :RunConfig

      

   .. attribute:: data
      :annotation: :WMTQEDataset.Config

      

   .. attribute:: system
      :annotation: :QESystem.Config

      

   .. attribute:: use_gpu
      :annotation: :bool = False

      If true and only if available, use the CUDA device specified in ``gpu_id`` or the
      first CUDA device. Otherwise, use the CPU.


   .. attribute:: gpu_id
      :annotation: :Optional[int]

      Use CUDA on the listed device, only if ``use_gpu`` is true.


   .. attribute:: verbose
      :annotation: :bool = False

      

   .. attribute:: quiet
      :annotation: :bool = False

      

   .. method:: enforce_loading(cls, v)


   .. method:: setup_gpu(cls, v)


   .. method:: setup_gpu_id(cls, v, values)



.. function:: load_system(system_path: Union[str, Path], gpu_id: Optional[int] = None)

   Load a pretrained system (model) into a `Runner` object.

   :param system_path: A path to the saved checkpoint file produced by a training run.
   :param gpu_id: id of the gpu to load the model into (-1 or None to use CPU)

   Throws:
     Exception: If the path does not exist, or is not a valid system file.


.. function:: predict_from_configuration(configuration_dict: Dict[str, Any])

   Run the entire prediction pipeline using the configuration options received.


.. function:: run(config: Configuration, output_dir: Path) -> Tuple[Dict[str, List], Optional[MetricsReport]]

   Run the prediction pipeline.

   Load the model and necessary files and create the model's predictions for the
   configured data partition.

   :param config: validated configuration values for the (predict) pipeline.
   :param output_dir: directory where to save predictions.

   :returns: Dictionary with format {'target': predictions}
   :rtype: Predictions


.. function:: make_predictions(output_dir: Path, best_model_path: Path, data_partition: Literal['train', 'valid', 'test'], data_config: WMTQEDataset.Config, outputs_config: QEOutputs.Config = None, batch_size: Union[int, BatchSizeConfig] = None, num_workers: int = 0, gpu_id: int = None)

   Make predictions over the validation set using the best model created during
   training.

   :param output_dir: output Directory where predictions should be saved.
   :param best_model_path: path pointing to the checkpoint with best performance.
   :param data_partition: on which dataset to predict (one of 'train', 'valid', 'test').
   :param data_config: configuration containing options for the ``data_partition`` set.
   :param outputs_config: configuration specifying which outputs to activate.
   :param batch_size: for predicting.
   :param num_workers: number of parallel data loaders.
   :param gpu_id: GPU to use for predicting; 0 for CPU.

   :returns: predictions}.
   :rtype: dictionary with predictions in the format {'target'


.. function:: setup_run(config: RunConfig, quiet=False, debug=False, anchor_dir: Path = None) -> Path

   Prepare for running the prediction pipeline.

   This includes setting up the output directory, random seeds, and loggers.

   :param config: configuration options.
   :param quiet: whether to suppress info log messages.
   :param debug: whether to additionally log debug messages
                 (:param:`quiet` has precedence)
   :param anchor_dir: directory to use as root for paths.

   :returns: the resolved path to the output directory.


