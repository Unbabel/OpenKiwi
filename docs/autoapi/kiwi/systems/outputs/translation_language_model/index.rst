:mod:`kiwi.systems.outputs.translation_language_model`
======================================================

.. py:module:: kiwi.systems.outputs.translation_language_model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.outputs.translation_language_model.MaskedWordOutput
   kiwi.systems.outputs.translation_language_model.TLMOutputs



.. py:class:: MaskedWordOutput(input_size, pad_idx, start_idx, stop_idx)

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

   .. method:: forward(self, features_tensor)



.. py:class:: TLMOutputs(inputs_dims: Dict[str, int], vocabs: Dict[str, Vocabulary], config: Config, pretraining: bool = False)

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

      .. attribute:: fine_tune
         :annotation: :bool = False

         Continue training an encoder on the post-edited text.
         Recommended if you have access to PE.
         Requires setting `system.data.train.input.pe`, `system.data.valid.input.pe`



   .. method:: forward(self, features, batch_inputs)


   .. method:: loss(self, model_out, batch_outputs)


   .. method:: metrics_step(self, batch, model_out, loss_dict)


   .. method:: metrics_end(self, steps, prefix='')


   .. method:: metrics(self)
      :property:


   .. method:: labels(self, field)



