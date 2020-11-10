:mod:`kiwi.systems.predictor_estimator`
=======================================

.. py:module:: kiwi.systems.predictor_estimator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.predictor_estimator.ModelConfig
   kiwi.systems.predictor_estimator.PredictorEstimator



.. data:: logger
   

   

.. py:class:: ModelConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: encoder
      :annotation: :PredictorEncoder.Config

      

   .. attribute:: decoder
      :annotation: :EstimatorDecoder.Config

      

   .. attribute:: outputs
      :annotation: :QEOutputs.Config

      

   .. attribute:: tlm_outputs
      :annotation: :TLMOutputs.Config

      


.. py:class:: PredictorEstimator(config, data_config: WMTQEDataset.Config = None, module_dict: Dict[str, Any] = None)

   Bases: :class:`kiwi.systems.qe_system.QESystem`

   Predictor-Estimator QE model (proposed in 2017).

   .. py:class:: Config

      Bases: :class:`kiwi.systems.qe_system.QESystem.Config`

      System configuration base class.

      .. attribute:: model
         :annotation: :ModelConfig

         



