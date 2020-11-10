:mod:`kiwi.modules.common.layer_norm`
=====================================

.. py:module:: kiwi.modules.common.layer_norm


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.modules.common.layer_norm.TFLayerNorm



.. py:class:: TFLayerNorm(hidden_size, eps=1e-06)

   Bases: :class:`torch.nn.Module`

   Construct a layer normalization module with epsilon inside the
   square root (tensorflow style).

   This is equivalent to HuggingFace's BertLayerNorm module.

   .. method:: forward(self, x)



