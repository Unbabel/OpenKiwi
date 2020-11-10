:mod:`kiwi.data.vocabulary`
===========================

.. py:module:: kiwi.data.vocabulary


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.data.vocabulary.Vocabulary



.. data:: logger
   

   

.. py:class:: Vocabulary(counter, max_size=None, min_freq=1, unk_token=None, pad_token=None, bos_token=None, eos_token=None, specials=None, vectors=None, unk_init=None, vectors_cache=None, specials_first=True, rare_with_vectors=True, add_vectors_vocab=False)

   Define a vocabulary object that will be used to numericalize a field.

   .. attribute:: counter

      A collections.Counter object holding the frequencies of tokens in the
      data used to build the Vocab.

   .. attribute:: stoi

      A dictionary mapping token strings to numerical identifiers;
      NOTE: use :meth:`token_to_id` to do the conversion.

   .. attribute:: itos

      A list of token strings indexed by their numerical identifiers;
      NOTE: use :meth:`id_to_token` to do the conversion.

   .. method:: token_to_id(self, token)


   .. method:: id_to_token(self, idx)


   .. method:: pad_id(self)
      :property:


   .. method:: bos_id(self)
      :property:


   .. method:: eos_id(self)
      :property:


   .. method:: __len__(self)


   .. method:: net_length(self)


   .. method:: max_size(self, max_size)

      Limit the vocabulary size.

      The assumption here is that the vocabulary was created from a list of tokens
      sorted by descending frequency.


   .. method:: __getstate__(self)


   .. method:: __setstate__(self, state)



