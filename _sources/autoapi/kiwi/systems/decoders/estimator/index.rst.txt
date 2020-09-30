:mod:`kiwi.systems.decoders.estimator`
======================================

.. py:module:: kiwi.systems.decoders.estimator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.decoders.estimator.EstimatorDecoder



.. data:: logger
   

   

.. py:class:: EstimatorDecoder(inputs_dims, config)

   Bases: :class:`kiwi.systems._meta_module.MetaModule`

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

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      Base class for all pydantic configs. Used to configure base behaviour of configs.

      .. attribute:: hidden_size
         :annotation: :int = 100

         Size of hidden layers in LSTM


      .. attribute:: rnn_layers
         :annotation: :PositiveInt = 1

         Layers in PredictorEstimator RNN


      .. attribute:: use_mlp
         :annotation: :bool = True

         Pass the PredictorEstimator input through a linear layer reducing
         dimensionality before RNN.


      .. attribute:: dropout
         :annotation: :confloat(ge=0.0, le=1.0) = 0.0

         

      .. attribute:: use_v0_buggy_strategy
         :annotation: :bool = False

         The Predictor implementation in Kiwi<=0.3.4 had a bug in applying the LSTM
         to encode source (it used lengths too short by 2) and in reversing the target
         embeddings for applying the backward LSTM (also short by 2). This flag is set
         to true when loading a saved model from those versions.


      .. method:: dropout_on_rnns(cls, v, values)



   .. method:: size(self, field=None)


   .. method:: forward(self, features, batch_inputs)



