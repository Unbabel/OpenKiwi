:mod:`kiwi.systems._meta_module`
================================

.. py:module:: kiwi.systems._meta_module


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.systems._meta_module.Serializable
   kiwi.systems._meta_module.MetaModule



.. data:: logger
   

   

.. py:class:: Serializable

   .. attribute:: subclasses
      

      

   .. method:: register_subclass(cls, subclass)
      :classmethod:


   .. method:: retrieve_subclass(cls, subclass_name)
      :classmethod:


   .. method:: load(cls, path)
      :classmethod:


   .. method:: save(self, path)


   .. method:: from_dict(cls, *args, **kwargs)
      :classmethod:
      :abstractmethod:


   .. method:: to_dict(cls, include_state=True)
      :classmethod:
      :abstractmethod:



.. py:class:: MetaModule(config: Config)

   Bases: :class:`torch.nn.Module`, :class:`kiwi.systems._meta_module.Serializable`

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


   .. method:: from_dict(cls, module_dict, **kwargs)
      :classmethod:


   .. method:: to_dict(self, include_state=True)



