:mod:`kiwi.modules.common.scorer`
=================================

.. py:module:: kiwi.modules.common.scorer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.modules.common.scorer.Scorer
   kiwi.modules.common.scorer.MLPScorer



.. py:class:: Scorer(scaled: bool = True)

   Bases: :class:`torch.nn.Module`

   Score function for attention module.

   :param scaled: whether to scale scores by `sqrt(hidden_size)` as proposed by the
                  "Attention is All You Need" paper.

   .. method:: scale(self, hidden_size: int) -> float

      Denominator for scaling the scores.

      :param hidden_size: max hidden size between query and keys.

      :returns: sqrt(hidden_size) if `scaled` is True, 1 otherwise.


   .. method:: forward(self, query: torch.FloatTensor, keys: torch.FloatTensor) -> torch.FloatTensor
      :abstractmethod:

      Compute scores for each key of size n given the queries of size m.

      The three dots (...) represent any other dimensions, such as the
      number of heads (useful if you use a multi head attention).

      :param query: query matrix ``(bs, ..., target_len, m)``.
      :param keys: keys matrix ``(bs, ..., source_len, n)``.

      :returns: matrix representing scores between source words and target words
                ``(bs, ..., target_len, source_len)``



.. py:class:: MLPScorer(query_size, key_size, layer_sizes=None, activation=nn.Tanh, **kwargs)

   Bases: :class:`kiwi.modules.common.scorer.Scorer`

   MultiLayerPerceptron Scorer with variable nb of layers and neurons.

   .. method:: forward(self, query, keys)

      Compute scores for each key of size n given the queries of size m.

      The three dots (...) represent any other dimensions, such as the
      number of heads (useful if you use a multi head attention).

      :param query: query matrix ``(bs, ..., target_len, m)``.
      :param keys: keys matrix ``(bs, ..., source_len, n)``.

      :returns: matrix representing scores between source words and target words
                ``(bs, ..., target_len, source_len)``



