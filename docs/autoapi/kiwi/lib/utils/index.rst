:mod:`kiwi.lib.utils`
=====================

.. py:module:: kiwi.lib.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.lib.utils.configure_seed
   kiwi.lib.utils.configure_logging
   kiwi.lib.utils.save_config_to_file
   kiwi.lib.utils.setup_output_directory
   kiwi.lib.utils.file_to_configuration
   kiwi.lib.utils.arguments_to_configuration
   kiwi.lib.utils._reset_hydra


.. function:: configure_seed(seed: int)

   Configure the random seed for all relevant packages.

   These include: random, numpy, torch, torch.cuda and PYTHONHASHSEED.

   :param seed: the random seed to be set.


.. function:: configure_logging(output_dir: Path = None, verbose: bool = False, quiet: bool = False)

   Configure the output logger.

   Set up the log format, logging level, and output directory of logging.

   :param output_dir: the directory where log output will be stored; defaults to None.
   :param verbose: change logging level to debug.
   :param quiet: change logging level to warning to suppress info logs.


.. function:: save_config_to_file(config: BaseConfig, file_name: Union[str, Path])

   Save a configuration object to file.

   :param file_name: where to saved the configuration.
   :param config: a pydantic configuration object.


.. function:: setup_output_directory(output_dir, run_uuid=None, experiment_id=None, create=True) -> str

   Set up the output directory.

   This means either creating one, or verifying that the provided directory exists.
   Output directories are created using the run and experiment ids.

   :param output_dir: the target output directory.
   :param run_uuid: the hash of the current run.
   :param experiment_id: the id of the current experiment.
   :param create: whether to create the directory.

   :returns: the path to the resolved output directory.


.. function:: file_to_configuration(config_file: Union[str, Path]) -> Dict

   Utility function to handle converting a configuration file to
   a dictionary with the correct hydra composition.

   Creates an argument dict and calls `arguments_to_configuration`

   :param config_file: path to a configuration file.

   :returns: Dictionary of the configuration imported from config file.


.. function:: arguments_to_configuration(arguments: Dict) -> Dict

   Processes command line arguments into a dictionary.
   Handles hydra file composition and parameter overwrites.

   :param arguments: dictionary with all the cmd_line arguments passed to kiwi.

   :returns: Dictionary of the config imported from the config file.


.. function:: _reset_hydra()

   Utility function used to handle global hydra state


