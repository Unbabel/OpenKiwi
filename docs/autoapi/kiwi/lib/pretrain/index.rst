:mod:`kiwi.lib.pretrain`
========================

.. py:module:: kiwi.lib.pretrain


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.lib.pretrain.Configuration



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.lib.pretrain.pretrain_from_file
   kiwi.lib.pretrain.pretrain_from_configuration


.. data:: logger
   

   

.. py:class:: Configuration

   Bases: :class:`kiwi.lib.train.Configuration`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: system
      :annotation: :TLMSystem.Config

      


.. function:: pretrain_from_file(filename) -> TrainRunInfo

   Load options from a config file and call the pretraining procedure.

   :param filename: of the configuration file.

   :returns: object with training information.


.. function:: pretrain_from_configuration(configuration_dict) -> TrainRunInfo

   Run the entire training pipeline using the configuration options received.

   :param configuration_dict: dictionary with config options.

   :returns: object with training information.


