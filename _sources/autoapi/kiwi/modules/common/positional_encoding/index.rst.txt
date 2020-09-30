:mod:`kiwi.modules.common.positional_encoding`
==============================================

.. py:module:: kiwi.modules.common.positional_encoding


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.modules.common.positional_encoding.PositionalEncoding



.. py:class:: PositionalEncoding(max_seq_len: int, hidden_size: int)

   Bases: :class:`torch.nn.Module`

   Absolute positional encoding mechanism.

   :param max_seq_len: hypothetical maximum sequence length (usually 1000).
   :param hidden_size: embeddings size.

   .. method:: forward(self, emb)



.. data:: batch_size
   :annotation: = 8

   

