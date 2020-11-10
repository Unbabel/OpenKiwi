:mod:`kiwi.systems.decoders.linear`
===================================

.. py:module:: kiwi.systems.decoders.linear


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.decoders.linear.LinearDecoder



.. py:class:: LinearDecoder(inputs_dims, config)

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
         :annotation: :int = 250

         Size of hidden layer


      .. attribute:: dropout
         :annotation: :confloat(ge=0.0, le=1.0) = 0.0

         

      .. attribute:: bottleneck_size
         :annotation: :int = 100

         


   .. method:: size(self, field=None)


   .. method:: forward(self, features: Dict[str, torch.Tensor], batch_inputs: MultiFieldBatch)



