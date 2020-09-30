:mod:`kiwi.data.batch`
======================

.. py:module:: kiwi.data.batch


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.data.batch.BatchedSentence
   kiwi.data.batch.MultiFieldBatch



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.data.batch.tensors_to


.. py:class:: BatchedSentence

   .. attribute:: tensor
      :annotation: :torch.Tensor

      

   .. attribute:: lengths
      :annotation: :torch.Tensor

      

   .. attribute:: bounds
      :annotation: :torch.Tensor

      

   .. attribute:: bounds_lengths
      :annotation: :torch.Tensor

      

   .. attribute:: strict_masks
      :annotation: :torch.Tensor

      

   .. attribute:: number_of_tokens
      :annotation: :torch.Tensor

      

   .. method:: pin_memory(self)


   .. method:: to(self, *args, **kwargs)



.. py:class:: MultiFieldBatch(batch: dict)

   Bases: :class:`dict`

   dict() -> new empty dictionary
   dict(mapping) -> new dictionary initialized from a mapping object's
       (key, value) pairs
   dict(iterable) -> new dictionary initialized as if via:
       d = {}
       for k, v in iterable:
           d[k] = v
   dict(**kwargs) -> new dictionary initialized with the name=value pairs
       in the keyword argument list.  For example:  dict(one=1, two=2)

   .. method:: pin_memory(self)


   .. method:: to(self, *args, **kwargs)



.. function:: tensors_to(tensors, *args, **kwargs)


