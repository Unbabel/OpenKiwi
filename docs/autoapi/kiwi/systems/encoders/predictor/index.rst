:mod:`kiwi.systems.encoders.predictor`
======================================

.. py:module:: kiwi.systems.encoders.predictor


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.encoders.predictor.DualSequencesEncoder
   kiwi.systems.encoders.predictor.PredictorEncoder



.. data:: logger
   

   

.. py:class:: DualSequencesEncoder(input_size_a, input_size_b, hidden_size, output_size, num_layers, dropout, _use_v0_buggy_strategy=False)

   Bases: :class:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super(Model, self).__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. method:: forward(self, embeddings_a, lengths_a, mask_a, embeddings_b, lengths_b)


   .. method:: contextualize_b(self, embeddings, lengths, hidden)


   .. method:: encode_b(self, embeddings, forward_contexts, backward_contexts, contexts_a, attention_mask)

      Encode sequence B.

      Build a feature vector for each position i using left context i-1 and right
      context i+1. In the original implementation, this resulted in a returned tensor
      with -2 timesteps (dim=1). We have now changed it to return the same number
      of timesteps as the input. The consequence is that callers now have to deal
      with BOS and EOS in a different way, but hopefully this new behaviour is more
      consistent and less surprising. The old behaviour can be forced by setting
      ``self._use_v0_buggy_strategy`` to True.


   .. method:: _reverse_padded_seq(lengths, sequence)
      :staticmethod:

      Reverse a batch of padded sequences of different length.


   .. method:: _split_hidden(hidden)
      :staticmethod:

      Split hidden state into forward/backward parts.



.. py:class:: PredictorEncoder(vocabs: Dict[str, Vocabulary], config: Config, pretraining: bool = False, pre_load_model: bool = True)

   Bases: :class:`kiwi.systems._meta_module.MetaModule`

   Bidirectional Conditional Language Model

   Implemented after Kim et al 2017, see: http://www.statmt.org/wmt17/pdf/WMT63.pdf

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      Base class for all pydantic configs. Used to configure base behaviour of configs.

      .. attribute:: encode_source
         :annotation: :bool = False

         

      .. attribute:: hidden_size
         :annotation: :int = 400

         Size of hidden layers in LSTM.


      .. attribute:: rnn_layers
         :annotation: :int = 3

         Number of RNN layers in the Predictor.


      .. attribute:: dropout
         :annotation: :float = 0.0

         

      .. attribute:: share_embeddings
         :annotation: :bool = False

         Tie input and output embeddings for target.


      .. attribute:: out_embeddings_dim
         :annotation: :Optional[int]

         Word Embedding in Output layer.


      .. attribute:: use_mismatch_features
         :annotation: :bool = False

         Whether to use Alibaba's mismatch features.


      .. attribute:: embeddings
         :annotation: :InputEmbeddingsConfig

         

      .. attribute:: use_v0_buggy_strategy
         :annotation: :bool = False

         The Predictor implementation in Kiwi<=0.3.4 had a bug in applying the LSTM
         to encode source (it used lengths too short by 2) and in reversing the target
         embeddings for applying the backward LSTM (also short by 2). This flag is set
         to true when loading a saved model from those versions.


      .. attribute:: v0_start_stop
         :annotation: :bool = False

         Whether pre_qe_f_v is padded on both ends or
         post_qe_f_v is strip on both ends.


      .. method:: dropout_on_rnns(cls, v, values)


      .. method:: no_implementation(cls, v)



   .. method:: input_data_encoders(cls, config: Config)
      :classmethod:


   .. method:: size(self, field=None)


   .. method:: forward(self, batch_inputs, include_target_logits=False)



