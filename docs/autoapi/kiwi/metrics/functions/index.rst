:mod:`kiwi.metrics.functions`
=============================

.. py:module:: kiwi.metrics.functions


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.metrics.functions.mean_absolute_error
   kiwi.metrics.functions.mean_squared_error
   kiwi.metrics.functions.delta_average
   kiwi.metrics.functions.precision
   kiwi.metrics.functions.recall
   kiwi.metrics.functions.fscore
   kiwi.metrics.functions.confusion_matrix
   kiwi.metrics.functions.scores_for_class
   kiwi.metrics.functions.precision_recall_fscore_support
   kiwi.metrics.functions.f1_product
   kiwi.metrics.functions.f1_scores
   kiwi.metrics.functions.matthews_correlation_coefficient


.. function:: mean_absolute_error(y, y_hat)


.. function:: mean_squared_error(y, y_hat)


.. function:: delta_average(y_true, y_rank) -> float

   Calculate the DeltaAvg score.

   This is a much faster version than the Perl one provided in the WMT QE task 1.

   References: could not find any.

   Author: Fabio Kepler (contributed to MARMOT).

   :param y_true: array of reference score (not rank) of each segment.
   :param y_rank: array of rank of each segment.

   :returns: the absolute delta average score.


.. function:: precision(tp, fp, fn)


.. function:: recall(tp, fp, fn)


.. function:: fscore(tp, fp, fn)


.. function:: confusion_matrix(hat_y, y, n_classes=None)


.. function:: scores_for_class(class_index, cnfm)


.. function:: precision_recall_fscore_support(hat_y, y, labels=None)


.. function:: f1_product(hat_y, y)


.. function:: f1_scores(hat_y, y) -> Tuple[Any, np.ndarray]

   Compute and return f1 for each class and the f1_product.


.. function:: matthews_correlation_coefficient(hat_y, y)

   Compute Matthews Correlation Coefficient.

   :param hat_y: list of np array of predicted binary labels.
   :param y: list of np array of true binary labels.

   :returns: the Matthews correlation coefficient of hat_y and y.


