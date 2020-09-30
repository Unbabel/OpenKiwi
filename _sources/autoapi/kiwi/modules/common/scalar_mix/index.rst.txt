:mod:`kiwi.modules.common.scalar_mix`
=====================================

.. py:module:: kiwi.modules.common.scalar_mix


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.modules.common.scalar_mix.ScalarMixWithDropout



.. py:class:: ScalarMixWithDropout(mixture_size: int, do_layer_norm: bool = False, initial_scalar_parameters: list = None, trainable: bool = True, dropout: float = None, dropout_value: float = -1e+20)

   Bases: :class:`torch.nn.Module`

   Compute a parameterised scalar mixture of N tensors.

   :math:`mixture = \gamma * \sum(s_k * tensor_k)`,
   where :math:`s = softmax(w)`, with :math:`w` and :math:`gamma` scalar parameters.

   If ``do_layer_norm=True``, then apply layer normalization to each tensor before
   weighting.

   If ``dropout > 0``, then for each scalar weight, adjust its softmax weight mass to 0
   with the dropout probability (i.e., setting the unnormalized weight to -inf).
   This effectively should redistribute dropped probability mass to all other weights.

   Original implementation:
       - https://github.com/Hyperparticle/udify
   Copied from COMET:
       - https://gitlab.com/Unbabel/language-technologies/unbabel-comet

   .. method:: forward(self, tensors: list, mask: torch.Tensor = None) -> torch.Tensor

      Compute a weighted average of the 'tensors'.

      The input tensors can be any shape with at least two dimensions, but must all
      have the same shape.

      When ``do_layer_norm=True``, ``mask`` is required. If ``tensors`` have
      dimensions ``(dim_0, ..., dim_{n-1}, dim_n)``, then ``mask`` should have dims
      ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
      ``(batch_size, timesteps, dim)`` and ``mask`` of shape
      ``(batch_size, timesteps)``.



