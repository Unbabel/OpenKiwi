:mod:`kiwi.data.encoders.wmt_qe_data_encoder`
=============================================

.. py:module:: kiwi.data.encoders.wmt_qe_data_encoder


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.data.encoders.wmt_qe_data_encoder.InputFields
   kiwi.data.encoders.wmt_qe_data_encoder.EmbeddingsConfig
   kiwi.data.encoders.wmt_qe_data_encoder.VocabularyConfig
   kiwi.data.encoders.wmt_qe_data_encoder.WMTQEDataEncoder



.. data:: logger
   

   

.. data:: T
   

   

.. py:class:: InputFields

   Bases: :class:`pydantic.generics.GenericModel`, :class:`Generic[T]`

   .. attribute:: source
      :annotation: :T

      

   .. attribute:: target
      :annotation: :T

      


.. py:class:: EmbeddingsConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Paths to word embeddings file for each input field.

   .. attribute:: source
      :annotation: :Optional[Path]

      

   .. attribute:: target
      :annotation: :Optional[Path]

      

   .. attribute:: post_edit
      :annotation: :Optional[Path]

      

   .. attribute:: source_pos
      :annotation: :Optional[Path]

      

   .. attribute:: target_pos
      :annotation: :Optional[Path]

      

   .. attribute:: format
      :annotation: :Literal['polyglot', 'word2vec', 'fasttext', 'glove', 'text'] = polyglot

      Word embeddings format. See README for specific formatting instructions.



.. py:class:: VocabularyConfig

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: min_frequency
      :annotation: :InputFields[PositiveInt] = 1

      Only add to vocabulary words that occur more than this number of times in the
      training dataset (doesn't apply to loaded or pretrained vocabularies).


   .. attribute:: max_size
      :annotation: :InputFields[Optional[PositiveInt]]

      Only create vocabulary with up to this many words (doesn't apply to loaded or
      pretrained vocabularies).


   .. attribute:: keep_rare_words_with_embeddings
      :annotation: = False

      Keep words that occur less then min-frequency but are
      in embeddings vocabulary.


   .. attribute:: add_embeddings_vocab
      :annotation: = False

      Add words from embeddings vocabulary to source/target vocabulary.


   .. method:: check_nested_options(cls, v)



.. py:class:: WMTQEDataEncoder(config: Config, field_encoders: Dict[str, TextEncoder] = None)

   Bases: :class:`kiwi.data.encoders.base.DataEncoders`

   .. py:class:: Config

      Bases: :class:`kiwi.utils.io.BaseConfig`

      Base class for all pydantic configs. Used to configure base behaviour of configs.

      .. attribute:: share_input_fields_encoders
         :annotation: :bool = False

         Share encoding/vocabs between source and target fields.


      .. attribute:: vocab
         :annotation: :VocabularyConfig

         

      .. attribute:: embeddings
         :annotation: :Optional[EmbeddingsConfig]

         

      .. method:: warn_missing_feature(cls, v)



   .. method:: fit_vocabularies(self, dataset: WMTQEDataset)


   .. method:: load_vocabularies(self, load_vocabs_from: Path = None, overwrite: bool = False)

      Load serialized Vocabularies from disk into fields.


   .. method:: vocabularies_from_dict(self, vocabs_dict: Dict, overwrite: bool = False)


   .. method:: vocabularies(self)
      :property:

      Return the vocabularies for all encoders that have one.

      :returns: A dict mapping encoder names to Vocabulary instances.


   .. method:: collate_fn(self, samples, device=None)



