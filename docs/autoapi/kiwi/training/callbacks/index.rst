:mod:`kiwi.training.callbacks`
==============================

.. py:module:: kiwi.training.callbacks


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.training.callbacks.BestMetricsInfo



.. data:: logger
   

   

.. py:class:: BestMetricsInfo(monitor: str = 'val_loss', min_delta: float = 0.0, verbose: bool = True, mode: str = 'auto')

   Bases: :class:`pytorch_lightning.Callback`

   Class for logging current training metrics along with the best so far.

   .. method:: on_train_begin(self, trainer, pl_module)


   .. method:: on_train_end(self, trainer, pl_module)

      Called when the train ends.


   .. method:: on_validation_end(self, trainer, pl_module)

      Called when the validation loop ends.



