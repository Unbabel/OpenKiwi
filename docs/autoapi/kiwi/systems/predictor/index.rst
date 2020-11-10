:mod:`kiwi.systems.predictor`
=============================

.. py:module:: kiwi.systems.predictor


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.predictor.ModelConfig
   kiwi.systems.predictor.Predictor



.. data:: logger
   

   

.. py:class:: ModelConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: encoder
      :annotation: :PredictorEncoder.Config

      

   .. attribute:: tlm_outputs
      :annotation: :TLMOutputs.Config

      


.. py:class:: Predictor(config: Config, data_config: WMTQEDataset.Config = None, module_dict: Dict[str, Any] = None)

   Bases: :class:`kiwi.systems.tlm_system.TLMSystem`

   Predictor TLM, used for the Predictor-Estimator QE model (proposed in 2017).

   .. py:class:: Config

      Bases: :class:`kiwi.systems.tlm_system.TLMSystem.Config`

      System configuration base class.

      .. attribute:: model
         :annotation: :ModelConfig

         



