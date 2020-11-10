:mod:`kiwi.runner`
==================

.. py:module:: kiwi.runner


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.runner.Predictions
   kiwi.runner.Runner



.. data:: logger
   

   

.. py:class:: Predictions

   .. attribute:: sentences_hter
      :annotation: :List[float]

      

   .. attribute:: target_tags_BAD_probabilities
      :annotation: :List[List[float]]

      

   .. attribute:: target_tags_labels
      :annotation: :List[List[str]]

      

   .. attribute:: source_tags_BAD_probabilities
      :annotation: :List[List[float]]

      

   .. attribute:: source_tags_labels
      :annotation: :List[List[str]]

      

   .. attribute:: gap_tags_BAD_probabilities
      :annotation: :List[List[float]]

      

   .. attribute:: gap_tags_labels
      :annotation: :List[List[str]]

      


.. py:class:: Runner(system: QESystem, device: Optional[int] = None)

   .. method:: configure_outputs(self, output_config: QEOutputs.Config)


   .. method:: wrap_predictions(predictions: Dict[str, List[Any]]) -> Predictions
      :staticmethod:


   .. method:: predict(self, source: List[str], target: List[str], alignments: List[str] = None, batch_size: int = 1, num_data_workers: int = 0) -> Predictions

      Create predictions for a list of examples.

      :param source: a list of sentences on a source language.
      :param target: a list of (translated) sentences on a target language.
      :param alignments: optional list of source-target alignments required only by the
                         NuQE model.
      :param batch_size: how large to build a batch (default: 1).
      :param num_data_workers: how many subprocesses to use for data loading.

      :returns: A ``Predictions`` object with predicted outputs for each example in the
                inputs. If input ``source`` and ``target`` are all empty, returned object
                has all attributes as ``None``. If there are aligned empty sentences at
                both ``source`` and ``target``, the corresponding returned prediction will
                contain empty/zero values (empty list for word level outputs, 0.0 for
                sentence level outputs).

      :raises Exception: If an example has an empty string as `source` xor `target`

          field (not both at the same time).

      .. rubric:: Notes

      ``source`` and ``target`` lenghts must match.

      .. rubric:: Example

      >>> from kiwi.lib import predict
      >>> runner = predict.load_system('../tests/toy-data/models/nuqe.ckpt')
      >>> source = ['a b c', 'd e f g']
      >>> target = ['q w e r', 't y']
      >>> alignments = ['0-0 1-1 1-2', '1-1 3-0']
      >>> predictions = runner.predict(source, target, alignments)
      >>> predictions.target_tags_BAD_probabilities  # doctest: +ELLIPSIS
      [[0.49699464440345764, 0.49956727027893066, ...], [..., 0.5013138651847839]]

      Predictions(
             sentences_hter=[0.2668147683143616, 0.26675286889076233],
             target_tags_BAD_probabilities=[
                 [
                     0.49699464440345764,
                     0.49956727027893066,
                     0.5025501847267151,
                     0.5057167410850525,
                 ],
                 [0.4967852830886841, 0.5013138651847839],
             ],
             target_tags_labels=[['OK', 'OK', 'BAD', 'BAD'], ['OK', 'BAD']],
             source_tags_BAD_probabilities=None,
             source_tags_labels=None,
             gap_tags_BAD_probabilities=[
                 [
                     0.42644527554512024,
                     0.42096763849258423,
                     0.41709718108177185,
                     0.4157106280326843,
                     0.41496342420578003,
                 ],
                 [0.42876192927360535, 0.4251120686531067, 0.4210476577281952],
             ],
             gap_tags_labels=[['OK', 'OK', 'OK', 'OK', 'OK'], ['OK', 'OK', 'OK']],
      )


   .. method:: run(self, iterator=None) -> Dict[str, List]


   .. method:: remove_empty_sentences(columns: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[int]]
      :staticmethod:


   .. method:: insert_dummy_outputs_for_empty_sentences(predictions: Dict[str, List], indices_of_empties: List[int]) -> Dict[str, List]
      :staticmethod:



