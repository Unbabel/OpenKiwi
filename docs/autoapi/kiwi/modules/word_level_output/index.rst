:mod:`kiwi.modules.word_level_output`
=====================================

.. py:module:: kiwi.modules.word_level_output


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.modules.word_level_output.WordLevelOutput
   kiwi.modules.word_level_output.GapTagsOutput



.. py:class:: WordLevelOutput(input_size, output_size, pad_idx, class_weights=None, remove_first=False, remove_last=False)

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

   .. method:: forward(self, features_tensor, batch_inputs=None)



.. py:class:: GapTagsOutput(input_size, output_size, pad_idx, class_weights=None, remove_first=False, remove_last=False)

   Bases: :class:`kiwi.modules.word_level_output.WordLevelOutput`

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

   .. method:: forward(self, features_tensor, batch_inputs=None)



