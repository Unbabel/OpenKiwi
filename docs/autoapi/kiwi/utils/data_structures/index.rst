:mod:`kiwi.utils.data_structures`
=================================

.. py:module:: kiwi.utils.data_structures


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.utils.data_structures.DefaultFrozenDict



.. py:class:: DefaultFrozenDict(mapping=None, default_key=const.UNK)

   Bases: :class:`collections.OrderedDict`

   dict() -> new empty dictionary
   dict(mapping) -> new dictionary initialized from a mapping object's
       (key, value) pairs
   dict(iterable) -> new dictionary initialized as if via:
       d = {}
       for k, v in iterable:
           d[k] = v
   dict(**kwargs) -> new dictionary initialized with the name=value pairs
       in the keyword argument list.  For example:  dict(one=1, two=2)

   .. method:: __getitem__(self, k)

      x.__getitem__(y) <==> x[y]



