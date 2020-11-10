:mod:`kiwi.systems.encoders.bert`
=================================

.. py:module:: kiwi.systems.encoders.bert


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.encoders.bert.TransformersTextEncoder
   kiwi.systems.encoders.bert.BertEncoder



.. data:: logger
   

   

.. py:class:: TransformersTextEncoder(tokenizer_name, is_source=False)

   Bases: :class:`kiwi.data.encoders.field_encoders.TextEncoder`

   Encode a field, handling vocabulary, tokenization and embeddings.

   Heavily inspired in torchtext and torchnlp.

   .. method:: fit_vocab(self, samples, vocab_size=None, vocab_min_freq=0, embeddings_name=None, keep_rare_words_with_embeddings=False, add_embeddings_vocab=False)



.. py:class:: BertEncoder(vocabs: Dict[str, Vocabulary], config: Config, pre_load_model: bool = True)

   Bases: :class:`kiwi.systems._meta_module.MetaModule`

   BERT model as presented in Google's paper and using Hugging Face's code

   .. rubric:: References

   https://arxiv.org/abs/1810.04805

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      Base class for all pydantic configs. Used to configure base behaviour of configs.

      .. attribute:: encode_source
         :annotation: :bool = False

         

      .. attribute:: model_name
         :annotation: :Union[str, Path] = bert-base-multilingual-cased

         Pre-trained BERT model to use.


      .. attribute:: use_mismatch_features
         :annotation: :bool = False

         Use Alibaba's mismatch features.


      .. attribute:: use_predictor_features
         :annotation: :bool = False

         Use features originally proposed in the Predictor model.


      .. attribute:: interleave_input
         :annotation: :bool = False

         Concatenate SOURCE and TARGET without internal padding
         (111222000 instead of 111002220)


      .. attribute:: freeze
         :annotation: :bool = False

         Freeze BERT during training.


      .. attribute:: use_mlp
         :annotation: :bool = True

         Apply a linear layer on top of BERT.


      .. attribute:: hidden_size
         :annotation: :int = 100

         Size of the linear layer on top of BERT.


      .. attribute:: scalar_mix_dropout
         :annotation: :confloat(ge=0.0, le=1.0) = 0.1

         

      .. attribute:: scalar_mix_layer_norm
         :annotation: :bool = True

         

      .. method:: fix_relative_path(cls, v)


      .. method:: no_implementation(cls, v)



   .. method:: load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True)

      Copies parameters and buffers from :attr:`state_dict` into
      this module and its descendants. If :attr:`strict` is ``True``, then
      the keys of :attr:`state_dict` must exactly match the keys returned
      by this module's :meth:`~torch.nn.Module.state_dict` function.

      :param state_dict: a dict containing parameters and
                         persistent buffers.
      :type state_dict: dict
      :param strict: whether to strictly enforce that the keys
                     in :attr:`state_dict` match the keys returned by this module's
                     :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
      :type strict: bool, optional

      :returns:     * **missing_keys** is a list of str containing the missing keys
                    * **unexpected_keys** is a list of str containing the unexpected keys
      :rtype: ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields


   .. method:: input_data_encoders(cls, config: Config)
      :classmethod:


   .. method:: size(self, field=None)


   .. method:: forward(self, batch_inputs, *args, include_target_logits=False, include_source_logits=False)


   .. method:: concat_input(source_batch, target_batch, pad_id)
      :staticmethod:

      Concatenate the target + source embeddings into one tensor.

      :returns:

                concatenation of embeddings, mask of target (as ones) and source
                    (as zeroes) and concatenation of attention_mask


   .. method:: split_outputs(features: Tensor, batch_inputs: MultiFieldBatch, interleaved: bool = False) -> Dict[str, Tensor]
      :staticmethod:

      Split features back into sentences A and B.

      :param features: BERT's output: ``[CLS] target [SEP] source [SEP]``.
                       Shape of (bs, 1 + target_len + 1 + source_len + 1, 2)
      :param batch_inputs: the regular batch object, containing ``source`` and ``target``
                           batches
      :param interleaved: whether the concat strategy was interleaved

      :returns: dict of tensors for ``source`` and ``target``.


   .. method:: interleave_input(source_batch, target_batch, pad_id)
      :staticmethod:

      Interleave the source + target embeddings into one tensor.

      This means making the input as [batch, target [SEP] source].

      :returns:

                interleave of embds, mask of target (as zeroes) and source (as ones)
                    and concatenation of attention_mask.


   .. method:: get_mismatch_features(logits, target, pred)
      :staticmethod:



