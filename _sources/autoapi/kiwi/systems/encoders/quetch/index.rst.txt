:mod:`kiwi.systems.encoders.quetch`
===================================

.. py:module:: kiwi.systems.encoders.quetch


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems.encoders.quetch.InputEmbeddingsConfig
   kiwi.systems.encoders.quetch.QUETCHEncoder



.. data:: logger
   

   

.. py:class:: InputEmbeddingsConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Embeddings size for each input field, if they are not loaded.

   .. attribute:: source
      :annotation: :TokenEmbeddings.Config

      

   .. attribute:: target
      :annotation: :TokenEmbeddings.Config

      

   .. attribute:: source_pos
      :annotation: :Optional[TokenEmbeddings.Config]

      

   .. attribute:: target_pos
      :annotation: :Optional[TokenEmbeddings.Config]

      


.. py:class:: QUETCHEncoder(vocabs: Dict[str, Vocabulary], config: Config, pre_load_model: bool = True)

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

      .. attribute:: window_size
         :annotation: :int = 3

         Size of sliding window.


      .. attribute:: embeddings
         :annotation: :InputEmbeddingsConfig

         


   .. method:: input_data_encoders(cls, config: Config)
      :classmethod:


   .. method:: size(self, field=None)


   .. method:: forward(self, batch_inputs)



