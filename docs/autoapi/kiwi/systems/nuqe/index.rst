:mod:`kiwi.systems.nuqe`
========================

.. py:module:: kiwi.systems.nuqe


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.nuqe.ModelConfig
   kiwi.systems.nuqe.NuQE



.. data:: logger
   

   

.. py:class:: ModelConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: encoder
      :annotation: :QUETCHEncoder.Config

      

   .. attribute:: decoder
      :annotation: :NuQEDecoder.Config

      

   .. attribute:: outputs
      :annotation: :QEOutputs.Config

      


.. py:class:: NuQE(config, data_config: WMTQEDataset.Config = None, module_dict: Dict[str, Any] = None)

   Bases: :class:`kiwi.systems.qe_system.QESystem`

   Neural Quality Estimation (NuQE) model for word level quality estimation.

   .. py:class:: Config

      Bases: :class:`kiwi.systems.qe_system.QESystem.Config`

      System configuration base class.

      .. attribute:: model
         :annotation: :ModelConfig

         



