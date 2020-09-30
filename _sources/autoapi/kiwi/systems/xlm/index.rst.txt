:mod:`kiwi.systems.xlm`
=======================

.. py:module:: kiwi.systems.xlm


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.xlm.ModelConfig
   kiwi.systems.xlm.XLM



.. data:: logger
   

   

.. py:class:: ModelConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: encoder
      :annotation: :XLMEncoder.Config

      

   .. attribute:: decoder
      :annotation: :LinearDecoder.Config

      

   .. attribute:: outputs
      :annotation: :QEOutputs.Config

      

   .. attribute:: tlm_outputs
      :annotation: :TLMOutputs.Config

      


.. py:class:: XLM(config, data_config: WMTQEDataset.Config = None, module_dict: Dict[str, Any] = None)

   Bases: :class:`kiwi.systems.qe_system.QESystem`

   XLM-based model for word level quality estimation.

   .. py:class:: Config

      Bases: :class:`kiwi.systems.qe_system.QESystem.Config`

      System configuration base class.

      .. attribute:: model
         :annotation: :ModelConfig

         



