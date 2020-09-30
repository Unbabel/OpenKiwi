:mod:`kiwi.systems.decoders.nuqe`
=================================

.. py:module:: kiwi.systems.decoders.nuqe


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.decoders.nuqe.NuQETargetDecoder
   kiwi.systems.decoders.nuqe.NuQESourceDecoder
   kiwi.systems.decoders.nuqe.NuQEDecoder



.. py:class:: NuQETargetDecoder(input_dim, config)

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

      .. attribute:: hidden_sizes
         :annotation: :conlist(int, min_items=4, max_items=4) = [400, 200, 100, 50]

         

      .. attribute:: dropout
         :annotation: :confloat(ge=0.0, le=1.0) = 0.4

         


   .. method:: forward(self, features, batch_inputs)


   .. method:: size(self)



.. py:class:: NuQESourceDecoder(input_dim, config)

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

      .. attribute:: hidden_sizes
         :annotation: :conlist(int, min_items=4, max_items=4) = [400, 200, 100, 50]

         

      .. attribute:: dropout
         :annotation: :confloat(ge=0.0, le=1.0) = 0.4

         


   .. method:: forward(self, features, batch_inputs)


   .. method:: size(self)



.. py:class:: NuQEDecoder(inputs_dims, config: Config)

   Bases: :class:`kiwi.systems._meta_module.MetaModule`

   Neural Quality Estimation (NuQE) model for word level quality estimation.

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      Base class for all pydantic configs. Used to configure base behaviour of configs.

      .. attribute:: target
         :annotation: :NuQETargetDecoder.Config

         

      .. attribute:: source
         :annotation: :NuQESourceDecoder.Config

         


   .. method:: forward(self, features, batch_inputs)


   .. method:: size(self, field=None)



