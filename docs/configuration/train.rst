Training configuration
======================

There are training configuration files available in ``config/`` for all supported models.

As an example, here is the configuration for the ``PredictorEstimator`` model:

.. literalinclude:: ../../config/predictor_estimator.yaml
   :language: yaml


Configuration class
-------------------

Full API reference: :class:`kiwi.lib.train.Configuration`


.. autosummary:
   :toctree: stubs

..   kiwi.lib.train.Configuration
   kiwi.lib.train.RunConfig
   kiwi.lib.train.TrainerConfig
   kiwi.data.datasets.wmt_qe_dataset.WMTQEDataset.Config
   kiwi.systems.qe_system.QESystem.Config


.. autoclass:: kiwi.lib.train.Configuration
   :noindex:
