:mod:`kiwi.utils.io`
====================

.. py:module:: kiwi.utils.io


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.utils.io.BaseConfig



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.utils.io.default_map_location
   kiwi.utils.io.load_torch_file
   kiwi.utils.io.save_file
   kiwi.utils.io.save_predicted_probabilities
   kiwi.utils.io.read_file
   kiwi.utils.io.target_gaps_to_target
   kiwi.utils.io.target_gaps_to_gaps
   kiwi.utils.io.generate_slug


.. data:: logger
   

   

.. py:class:: BaseConfig

   Bases: :class:`pydantic.BaseModel`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. py:class:: Config

      .. attribute:: extra
         

         



.. function:: default_map_location(storage, loc)


.. function:: load_torch_file(file_path, map_location=None)


.. function:: save_file(file_path, data, token_sep=' ', example_sep='\n')


.. function:: save_predicted_probabilities(directory, predictions, prefix='')


.. function:: read_file(path)

   Read a file into a list of lists of words.


.. function:: target_gaps_to_target(batch)

   Extract target tags from wmt18 format file.


.. function:: target_gaps_to_gaps(batch)

   Extract gap tags from wmt18 format file.


.. function:: generate_slug(text, delimiter='-')

   Convert text to a normalized "slug" without whitespace.

   Borrowed from the nice https://humanfriendly.readthedocs.io, by Peter Odding.

   :param text: the original text, for example ``Some Random Text!``.
   :param delimiter: the delimiter to use for separating words
                     (defaults to the ``-`` character).

   :returns: the slug text, for example ``some-random-text``.

   :raises ~exceptions.ValueError: in an empty slug.


