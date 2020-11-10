:mod:`kiwi.modules.common.attention`
====================================

.. py:module:: kiwi.modules.common.attention


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.modules.common.attention.Attention



.. py:class:: Attention(scorer, dropout=0)

   Bases: :class:`torch.nn.Module`

   Generic Attention Implementation.

      1. Use `query` and `keys` to compute scores (energies)
      2. Apply softmax to get attention probabilities
      3. Perform a dot product between `values` and probabilites (outputs)

   :param scorer: a scorer object
   :type scorer: kiwi.modules.common.Scorer
   :param dropout: dropout rate after softmax (default: 0.)
   :type dropout: float

   .. method:: forward(self, query, keys, values=None, mask=None)

      Compute the attention between query, keys and values.

      :param query: set of query vectors with shape of
                    (batch_size, ..., target_len, hidden_size)
      :type query: torch.Tensor
      :param keys: set of keys vectors with shape of
                   (batch_size, ..., source_len, hidden_size)
      :type keys: torch.Tensor
      :param values: set of values vectors with
                     shape of: (batch_size, ..., source_len, hidden_size).
                     If None, keys are treated as values. Default: None
      :type values: torch.Tensor, optional
      :param mask: Tensor representing valid
                   positions. If None, all positions are considered valid.
                   Shape of (batch_size, target_len)
      :type mask: torch.ByteTensor, optional

      :returns:

                combination of values and attention probabilities.
                    Shape of (batch_size, ..., target_len, hidden_size)
                torch.Tensor: attention probabilities between query and keys.
                    Shape of (batch_size, ..., target_len, source_len)
      :rtype: torch.Tensor



