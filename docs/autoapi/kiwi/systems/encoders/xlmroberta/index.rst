:mod:`kiwi.systems.encoders.xlmroberta`
=======================================

.. py:module:: kiwi.systems.encoders.xlmroberta


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.encoders.xlmroberta.XLMRobertaTextEncoder
   kiwi.systems.encoders.xlmroberta.XLMRobertaEncoder



.. data:: logger
   

   

.. py:class:: XLMRobertaTextEncoder(tokenizer_name='xlm-roberta-base', is_source=False)

   Bases: :class:`kiwi.data.encoders.field_encoders.TextEncoder`

   Encode a field, handling vocabulary, tokenization and embeddings.

   Heavily inspired in torchtext and torchnlp.

   .. method:: fit_vocab(self, samples, vocab_size=None, vocab_min_freq=0, embeddings_name=None, keep_rare_words_with_embeddings=False, add_embeddings_vocab=False)



.. py:class:: XLMRobertaEncoder(vocabs: Dict[str, Vocabulary], config: Config, pre_load_model: bool = True)

   Bases: :class:`kiwi.systems._meta_module.MetaModule`

   XLM-RoBERTa model, using HuggingFace's implementation.

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      Base class for all pydantic configs. Used to configure base behaviour of configs.

      .. attribute:: encode_source
         :annotation: :bool = False

         

      .. attribute:: model_name
         :annotation: :Union[str, Path] = xlm-roberta-base

         Pre-trained XLMRoberta model to use.


      .. attribute:: interleave_input
         :annotation: :bool = False

         Concatenate SOURCE and TARGET without internal padding
         (111222000 instead of 111002220)


      .. attribute:: use_mlp
         :annotation: :bool = True

         Apply a linear layer on top of XLMRoberta.


      .. attribute:: hidden_size
         :annotation: :int = 100

         Size of the linear layer on top of XLMRoberta.


      .. attribute:: pooling
         :annotation: :Literal['first_token', 'mean', 'll_mean', 'mixed'] = mixed

         Type of pooling used to extract features from the encoder. Options are:
         first_token: CLS_token is used for sentence representation
         mean: Use avg pooling for sentence representation using scalar mixed layers
         ll_mean: Mean pool of only last layer embeddings
         mixed: Concat CLS token with mean_pool


      .. attribute:: scalar_mix_dropout
         :annotation: :confloat(ge=0.0, le=1.0) = 0.1

         

      .. attribute:: scalar_mix_layer_norm
         :annotation: :bool = True

         

      .. attribute:: freeze
         :annotation: :bool = False

         Freeze XLMRoberta during training.


      .. attribute:: freeze_for_number_of_steps
         :annotation: :int = 0

         Freeze XLMR during training for this number of steps.


      .. method:: fix_relative_path(cls, v)



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


   .. method:: _check_freezing(self)


   .. method:: forward(self, batch_inputs, *args, include_logits=False)


   .. method:: concat_input(source_batch, target_batch, pad_id)
      :staticmethod:

      Concatenate tensors of two batches into one tensor.

      :returns:

                the concatenation, a mask of types (a as zeroes and b as ones)
                    and concatenation of attention_mask.


   .. method:: split_outputs(features, batch_inputs, interleaved=False)
      :staticmethod:

      Split contexts to get tag_side outputs.

      :param features: XLMRoberta output: <s> target </s> </s> source </s>
                       Shape of (bs, 1 + target_len + 2 + source_len + 1, 2)
      :type features: tensor
      :param batch_inputs:
      :param interleaved: whether the concat strategy was 'interleaved'.
      :type interleaved: bool

      :returns: dict of tensors, one per tag side.


   .. method:: interleave_input(source_batch, target_batch, pad_id)
      :staticmethod:

      Interleave the source + target embeddings into one tensor.

      This means making the input as [batch, target [SEP] source].

      :returns:

                interleave of embds, mask of target (as zeroes) and source (as ones)
                    and concatenation of attention_mask


   .. method:: get_mismatch_features(logits, target, pred)
      :staticmethod:



