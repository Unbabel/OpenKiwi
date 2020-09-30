:mod:`kiwi.systems.xlmroberta`
==============================

.. py:module:: kiwi.systems.xlmroberta


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.xlmroberta.ModelConfig
   kiwi.systems.xlmroberta.XLMRoberta



.. data:: logger
   

   

.. py:class:: ModelConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: encoder
      :annotation: :XLMRobertaEncoder.Config

      

   .. attribute:: decoder
      :annotation: :LinearDecoder.Config

      

   .. attribute:: outputs
      :annotation: :QEOutputs.Config

      

   .. attribute:: tlm_outputs
      :annotation: :TLMOutputs.Config

      


.. py:class:: XLMRoberta(config, data_config: WMTQEDataset.Config = None, module_dict: Dict[str, Any] = None)

   Bases: :class:`kiwi.systems.qe_system.QESystem`

   XLMRoberta-based Predictor-Estimator model

   .. py:class:: Config

      Bases: :class:`kiwi.systems.qe_system.QESystem.Config`

      System configuration base class.

      .. attribute:: model
         :annotation: :ModelConfig

         



