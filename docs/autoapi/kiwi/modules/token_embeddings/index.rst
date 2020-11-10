:mod:`kiwi.modules.token_embeddings`
====================================

.. py:module:: kiwi.modules.token_embeddings


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.modules.token_embeddings.TokenEmbeddings



.. py:class:: TokenEmbeddings(num_embeddings: int, pad_idx: int, config: Config, vectors=None)

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

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      Base class for all pydantic configs. Used to configure base behaviour of configs.

      .. attribute:: dim
         :annotation: :int = 50

         

      .. attribute:: freeze
         :annotation: :bool = False

         

      .. attribute:: dropout
         :annotation: :float = 0.0

         

      .. attribute:: use_position_embeddings
         :annotation: :bool = False

         

      .. attribute:: max_position_embeddings
         :annotation: :int = 4000

         

      .. attribute:: sparse_embeddings
         :annotation: :bool = False

         

      .. attribute:: scale_embeddings
         :annotation: :bool = False

         

      .. attribute:: input_layer_norm
         :annotation: :bool = False

         


   .. method:: num_embeddings(self)
      :property:


   .. method:: size(self)


   .. method:: forward(self, batch_input, *args)



