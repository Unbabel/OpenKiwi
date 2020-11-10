:mod:`kiwi.cli`
===============

.. py:module:: kiwi.cli

.. autoapi-nested-parse::

   Kiwi runner
   ~~~~~~~~~~~

   Quality Estimation toolkit.

   Invoke as ``kiwi PIPELINE``.

   Usage:
       kiwi [options] (train|pretrain|predict|evaluate) CONFIG_FILE [OVERWRITES ...]
       kiwi (train|pretrain|predict|evaluate) --example
       kiwi (-h | --help | --version)


   Pipelines:
       train          Train a QE model
       pretrain       Pretrain a TLM model to be used as an encoder for a QE model
       predict        Use a pre-trained model for prediction
       evaluate       Evaluate a model's predictions using popular metrics

   Disabled pipelines:
       search         Search training hyper-parameters for a QE model
       jackknife      Jackknife training data with model

   :param CONFIG_FILE    configuration file to use:
   :type CONFIG_FILE    configuration file to use: e.g., config/nuqe.yaml
   :param OVERWRITES     key=value to overwrite values in CONFIG_FILE; use ``key.subkey``: for nested keys.

   Options:
       -v --verbose          log debug messages
       -q --quiet            log only warning and error messages
       -h --help             show this help message and exit
       --version             show version and exit
       --example             print an example configuration file



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.cli.handle_example
   kiwi.cli.cli


.. function:: handle_example(arguments, caller)


.. function:: cli()


