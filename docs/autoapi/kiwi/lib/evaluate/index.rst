:mod:`kiwi.lib.evaluate`
========================

.. py:module:: kiwi.lib.evaluate


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.lib.evaluate.OutputConfig
   kiwi.lib.evaluate.Configuration
   kiwi.lib.evaluate.MetricsReport



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.lib.evaluate.evaluate_from_configuration
   kiwi.lib.evaluate.run
   kiwi.lib.evaluate.retrieve_gold_standard
   kiwi.lib.evaluate.normalize_prediction_files
   kiwi.lib.evaluate.split_wmt18_tags
   kiwi.lib.evaluate.read_sentence_scores_file
   kiwi.lib.evaluate.to_numeric_values
   kiwi.lib.evaluate.to_numeric_binary_labels
   kiwi.lib.evaluate.report_lengths_mismatch
   kiwi.lib.evaluate.lengths_match
   kiwi.lib.evaluate.word_level_scores
   kiwi.lib.evaluate.eval_word_level
   kiwi.lib.evaluate.sentence_level_scores
   kiwi.lib.evaluate.eval_sentence_level
   kiwi.lib.evaluate._extract_path_prefix


.. data:: logger
   

   

.. py:class:: OutputConfig

   Bases: :class:`kiwi.data.datasets.wmt_qe_dataset.OutputConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: gap_tags
      :annotation: :Optional[FilePath]

      Path to label file for gaps (only for predictions).


   .. attribute:: targetgaps_tags
      :annotation: :Optional[FilePath]

      Path to label file for target+gaps (only for predictions).



.. py:class:: Configuration

   Bases: :class:`kiwi.utils.io.BaseConfig`

   Base class for all pydantic configs. Used to configure base behaviour of configs.

   .. attribute:: gold_files
      :annotation: :wmt_qe_dataset.OutputConfig

      

   .. attribute:: predicted_files
      :annotation: :Optional[List[OutputConfig]]

      

   .. attribute:: predicted_dir
      :annotation: :Optional[List[Path]]

      One or more directories from where to read predicted files (using standard output
      names.


   .. method:: ensure_list(cls, v)


   .. method:: check_consistency(cls, v, values)



.. py:class:: MetricsReport

   .. attribute:: word_scores
      :annotation: :Dict[str, np.ndarray]

      

   .. attribute:: sentence_scores
      :annotation: :Dict[str, np.ndarray]

      

   .. method:: add_word_level_scores(self, name: str, scores: np.ndarray)


   .. method:: add_sentence_level_scores(self, name: str, scores: np.ndarray)


   .. method:: print_scores_table(self)


   .. method:: __str__(self)

      Return str(self).


   .. method:: _scores_str(scores: np.ndarray) -> str
      :staticmethod:



.. function:: evaluate_from_configuration(configuration_dict: Dict[str, Any])

   Evaluate a model's predictions based on the flags received from the configuration
   files.

   Refer to configuration for a list of available configuration flags for the evaluate
   pipeline.

   :param configuration_dict: options read from file or CLI


.. function:: run(config: Configuration) -> MetricsReport

   Runs the evaluation pipeline for evaluating a model's predictions. Essentially
   calculating metrics using `gold_targets` and `prediction_files`.

   Refer to configuration for a list of available options for this pipeline.

   :param config: Configuration Namespace

   :returns: Object with information for both word and sentence level metrics
   :rtype: MetricsReport


.. function:: retrieve_gold_standard(config: OutputConfig)


.. function:: normalize_prediction_files(predicted_files_config: List[OutputConfig], predicted_dir_config: List[Path])


.. function:: split_wmt18_tags(tags: List[List[Any]])

   Split tags list of lists in WMT18 format into target and gap tags.


.. function:: read_sentence_scores_file(sent_file)

   Read file with numeric scores for sentences.


.. function:: to_numeric_values(predictions: Union[str, List[str], List[List[str]]]) -> Union[int, float, List[int], List[float], List[List[int]], List[List[float]]]

   Convert text labels or string probabilities (for BAD) to int or float values,
   respectively.


.. function:: to_numeric_binary_labels(predictions: Union[str, float, List[str], List[List[str]], List[float], List[List[float]]], threshold: float = 0.5)

   Generate numeric labels from text labels or probabilities (for BAD).


.. function:: report_lengths_mismatch(gold, prediction)

   Checks if the number of gold and predictions labels match. Prints a warning and
   returns false if they do not.

   :param gold: list of gold labels
   :param prediction: list of predicted labels

   :returns: True if all lenghts match, False if not
   :rtype: bool


.. function:: lengths_match(gold, prediction)

   Checks if the number of gold and predictions labels match. Returns false if they
    do not.

   :param gold: list of gold labels
   :param prediction: list of predicted labels

   :returns: True if all lenghts match, False if not
   :rtype: bool


.. function:: word_level_scores(true_targets, predicted_targets, labels=const.LABELS)


.. function:: eval_word_level(true_targets, predictions: Dict[str, List[List[int]]]) -> np.ndarray


.. function:: sentence_level_scores(true_targets: List[float], predicted_targets: List[float]) -> Tuple[Tuple, Tuple]


.. function:: eval_sentence_level(true_targets, predictions: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]


.. function:: _extract_path_prefix(file_names)


