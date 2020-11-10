:mod:`kiwi.systems.encoders.xlm`
================================

.. py:module:: kiwi.systems.encoders.xlm


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.encoders.xlm.XLMTextEncoder
   kiwi.systems.encoders.xlm.XLMEncoder



.. data:: logger
   

   

.. py:class:: XLMTextEncoder(tokenizer_name)

   Bases: :class:`kiwi.data.encoders.field_encoders.TextEncoder`

   Encode a field, handling vocabulary, tokenization and embeddings.

   Heavily inspired in torchtext and torchnlp.

   .. method:: fit_vocab(self, samples, vocab_size=None, vocab_min_freq=0, embeddings_name=None, keep_rare_words_with_embeddings=False, add_embeddings_vocab=False)



.. py:class:: XLMEncoder(vocabs: Dict[str, Vocabulary], config: Config, pre_load_model: bool = True)

   Bases: :class:`kiwi.systems._meta_module.MetaModule`

   XLM model using Hugging Face's transformers library.

   The following command was used to fine-tune XLM on the in-domain data (retrieved
   from .pth file)::

       python train.py --exp_name tlm_clm --dump_path './dumped/'             --data_path '/mnt/shared/datasets/kiwi/parallel/en_de_indomain'             --lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh'             --clm_steps 'en-de,de-en' --mlm_steps 'en-de,de-en'             --reload_model 'models/mlm_tlm_xnli15_1024.pth' --encoder_only True             --emb_dim 1024 --n_layers 12 --n_heads 8 --dropout '0.1'             --attention_dropout '0.1' --gelu_activation true --batch_size 32             --bptt 256 --optimizer
           'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,weight_decay=0'             --epoch_size 200000 --validation_metrics _valid_mlm_ppl --max_vocab 95000             --tokens_per_batch 1200 --exp_id "5114"

   Old version was converted using hf-transformers util method::

       convert_xlm_checkpoint_to_pytorch(
           self.config.model_name / 'indomain.pth',
           self.config.model_name / 'finetuned_wmt_en-de'
       )

   Old settings in QE not really used for the best run and submission:

   .. code-block:: yaml

       fb-causal-lambda: 0.0
       fb-keep-prob: 0.1
       fb-mask-prob: 0.8
       fb-model: data/trained_models/fb_pretrain/xnli/indomain.pth
       fb-pred-prob: 0.15
       fb-rand-prob: 0.1
       fb-src-lang: en
       fb-tgt-lang: de
       fb-tlm-lambda: 0.0
       fb-vocab: data/trained_models/fb_pretrain/xnli/vocab_xnli_15.txt

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      Base class for all pydantic configs. Used to configure base behaviour of configs.

      .. attribute:: encode_source
         :annotation: :bool = False

         

      .. attribute:: model_name
         :annotation: :Union[str, Path] = xlm-mlm-tlm-xnli15-1024

         Pre-trained XLM model to use.


      .. attribute:: source_language
         :annotation: :str = en

         

      .. attribute:: target_language
         :annotation: :str = de

         

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

         Freeze XLM during training.


      .. attribute:: use_mlp
         :annotation: :bool = True

         Apply a linear layer on top of XLM.


      .. attribute:: hidden_size
         :annotation: :int = 100

         Size of the linear layer on top of XLM.


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


   .. method:: concat_input(batch_a, batch_b, pad_id, lang_a=None, lang_b=None)
      :staticmethod:

      Concatenate tensors of two batches into one tensor.

      :returns:

                the concatenation, a mask of types (a as zeroes and b as ones)
                    and concatenation of attention_mask.


   .. method:: interleave_input(batch_a, batch_b, pad_id, lang_a=None, lang_b=None)
      :staticmethod:

      Interleave the source + target embeddings into one tensor.

      This means making the input as [batch, target [SEP] source].

      :returns:

                interleave of embds, mask of target (as zeroes) and source (as ones)
                    and concatenation of attention_mask.


   .. method:: split_outputs(features: torch.Tensor, batch_inputs, interleaved: bool = False, label_a: str = const.SOURCE, label_b: str = const.TARGET)
      :staticmethod:

      Split contexts to get tag_side outputs.

      :param features: XLM output: <s> source </s> </s> target </s>
                       Shape of (bs, 1 + source_len + 2 + target_len + 1, 2)
      :type features: tensor
      :param batch_inputs:
      :param interleaved: whether the concat strategy was 'interleaved'.
      :type interleaved: bool
      :param label_a: dictionary key for sequence A in ``features``.
      :param label_b: dictionary key for sequence B in ``features``.

      :returns: dict of tensors, one per tag side.



