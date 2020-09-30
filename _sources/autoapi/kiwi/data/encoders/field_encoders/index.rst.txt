:mod:`kiwi.data.encoders.field_encoders`
========================================

.. py:module:: kiwi.data.encoders.field_encoders


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.data.encoders.field_encoders.TextEncoder
   kiwi.data.encoders.field_encoders.TagEncoder
   kiwi.data.encoders.field_encoders.InputEncoder
   kiwi.data.encoders.field_encoders.ScoreEncoder
   kiwi.data.encoders.field_encoders.BinaryScoreEncoder
   kiwi.data.encoders.field_encoders.AlignmentEncoder



.. data:: logger
   

   

.. py:class:: TextEncoder(tokenize=tokenizers.tokenize, detokenize=tokenizers.detokenize, subtokenize=None, pad_token=PAD, unk_token=UNK, bos_token=START, eos_token=STOP, unaligned_token=UNALIGNED, specials_first=True, include_lengths=True, include_bounds=True)

   Encode a field, handling vocabulary, tokenization and embeddings.

   Heavily inspired in torchtext and torchnlp.

   .. method:: fit_vocab(self, samples, vocab_size=None, vocab_min_freq=0, embeddings_name=None, keep_rare_words_with_embeddings=False, add_embeddings_vocab=False)


   .. method:: vocabulary(self)
      :property:


   .. method:: padding_index(self)
      :property:


   .. method:: encode(self, example)


   .. method:: batch_encode(self, iterator)



.. py:class:: TagEncoder(tokenize=tokenizers.tokenize, detokenize=tokenizers.detokenize, pad_token=PAD, include_lengths=True)

   Bases: :class:`kiwi.data.encoders.field_encoders.TextEncoder`

   Encode a field, handling vocabulary, tokenization and embeddings.

   Heavily inspired in torchtext and torchnlp.


.. py:class:: InputEncoder


.. py:class:: ScoreEncoder(dtype=torch.float)

   Bases: :class:`kiwi.data.encoders.field_encoders.InputEncoder`

   .. method:: encode(self, example)


   .. method:: batch_encode(self, iterator)



.. py:class:: BinaryScoreEncoder(dtype=torch.float)

   Bases: :class:`kiwi.data.encoders.field_encoders.ScoreEncoder`

   Transform HTER score into binary OK/BAD label.

   .. method:: encode(self, example)



.. py:class:: AlignmentEncoder(dtype=torch.int, account_for_bos_token=True, account_for_eos_token=True)

   Bases: :class:`kiwi.data.encoders.field_encoders.InputEncoder`

   .. method:: encode(self, example)


   .. method:: batch_encode(self, iterator)



