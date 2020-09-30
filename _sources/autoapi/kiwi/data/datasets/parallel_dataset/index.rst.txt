:mod:`kiwi.data.datasets.parallel_dataset`
==========================================

.. py:module:: kiwi.data.datasets.parallel_dataset


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.data.datasets.parallel_dataset.InputConfig
   kiwi.data.datasets.parallel_dataset.TrainingConfig
   kiwi.data.datasets.parallel_dataset.TestConfig
   kiwi.data.datasets.parallel_dataset.ParallelDataset



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.data.datasets.parallel_dataset.read_file


.. data:: logger
   

   

.. py:class:: InputConfig

   Bases: :class:`pydantic.BaseModel`

   .. attribute:: source
      :annotation: :FilePath

      Path to a corpus file in the source language


   .. attribute:: target
      :annotation: :FilePath

      Path to a corpus file in the target language



.. py:class:: TrainingConfig

   Bases: :class:`pydantic.BaseModel`

   .. attribute:: input
      :annotation: :InputConfig

      


.. py:class:: TestConfig

   Bases: :class:`pydantic.BaseModel`

   .. attribute:: input
      :annotation: :InputConfig

      


.. py:class:: ParallelDataset(columns: Dict[Any, Union[Iterable, List]])

   Bases: :class:`torch.utils.data.Dataset`

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs a index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.

   .. py:class:: Config

      Bases: :class:`pydantic.BaseModel`

      .. attribute:: buffer_size
         :annotation: :int

         Number of consecutive instances to be temporarily stored in
         the buffer, which will be used later for batching/bucketing.


      .. attribute:: train
         :annotation: :Optional[TrainingConfig]

         

      .. attribute:: valid
         :annotation: :Optional[TrainingConfig]

         

      .. attribute:: test
         :annotation: :Optional[TestConfig]

         

      .. attribute:: split
         :annotation: :Optional[confloat(gt=0.0, lt=1.0)]

         Split train dataset in case that no validation set is given.


      .. method:: ensure_there_is_validation_data(cls, v, values)



   .. method:: build(config: Config, directory=None, train=False, valid=False, test=False, split=0)
      :staticmethod:

      Build training, validation, and test datasets.

      :param config: configuration object with file paths and processing flags;
                     check out the docs for :class:`Config`.
      :param directory: if provided and paths in configuration are not absolute, use it
                        to anchor them.
      :param train: whether to build the training dataset.
      :param valid: whether to build the validation dataset.
      :param test: whether to build the testing dataset.
      :param split: If no validation set is provided, randomly sample
                    :math:`1-split` of training examples as validation set.
      :type split: float


   .. method:: __getitem__(self, index_or_field: Union[int, str]) -> Union[List[Any], Dict[str, Any]]

      Get a row with data from all fields or all rows for a given field


   .. method:: __len__(self)


   .. method:: __contains__(self, item)


   .. method:: sort_key(self, field='source')



.. function:: read_file(path, reader)


