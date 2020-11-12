:mod:`kiwi.metrics.metrics`
===========================

.. py:module:: kiwi.metrics.metrics


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.metrics.metrics.Metric
   kiwi.metrics.metrics.NLLMetric
   kiwi.metrics.metrics.LabeledMetric
   kiwi.metrics.metrics.CorrectMetric
   kiwi.metrics.metrics.F1MultMetric
   kiwi.metrics.metrics.MatthewsMetric
   kiwi.metrics.metrics.SentenceMetric
   kiwi.metrics.metrics.PearsonMetric
   kiwi.metrics.metrics.SpearmanMetric
   kiwi.metrics.metrics.RMSEMetric
   kiwi.metrics.metrics.PerplexityMetric
   kiwi.metrics.metrics.ExpectedErrorMetric



.. py:class:: Metric(*targets, prefix=None)

   .. attribute:: _name
      

      

   .. attribute:: best_ordering
      :annotation: = max

      

   .. method:: step(self, model_out, batch, losses)
      :abstractmethod:


   .. method:: compute(self, steps, prefix='')
      :abstractmethod:


   .. method:: name(self)
      :property:


   .. method:: num_tokens(self, batch, *targets)



.. py:class:: NLLMetric(*targets, prefix=None)

   Bases: :class:`kiwi.metrics.metrics.Metric`

   .. attribute:: _name
      :annotation: = NLL

      

   .. attribute:: best_ordering
      :annotation: = min

      

   .. method:: step(self, model_out, batch, losses)


   .. method:: compute(self, steps, prefix='')



.. py:class:: LabeledMetric(*args, labels=None, **kwargs)

   Bases: :class:`kiwi.metrics.metrics.Metric`, :class:`abc.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. method:: step(self, model_out, batch, losses)


   .. method:: get_target_flat(self, batch)


   .. method:: get_predictions_flat(self, model_out, batch)



.. py:class:: CorrectMetric(*args, labels=None, **kwargs)

   Bases: :class:`kiwi.metrics.metrics.LabeledMetric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. attribute:: _name
      :annotation: = CORRECT

      

   .. method:: step(self, model_out, batch, losses)


   .. method:: compute(self, steps, prefix='')



.. py:class:: F1MultMetric(*args, labels=None, **kwargs)

   Bases: :class:`kiwi.metrics.metrics.LabeledMetric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. attribute:: _name
      :annotation: = F1_MULT

      

   .. method:: compute(self, steps, prefix='')



.. py:class:: MatthewsMetric(*args, labels=None, **kwargs)

   Bases: :class:`kiwi.metrics.metrics.LabeledMetric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. attribute:: _name
      :annotation: = MCC

      

   .. method:: compute(self, steps, prefix='')



.. py:class:: SentenceMetric(*targets, prefix=None)

   Bases: :class:`kiwi.metrics.metrics.Metric`, :class:`abc.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. method:: step(self, model_out, batch, losses)



.. py:class:: PearsonMetric(*targets, prefix=None)

   Bases: :class:`kiwi.metrics.metrics.SentenceMetric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. attribute:: _name
      :annotation: = PEARSON

      

   .. method:: compute(self, steps, prefix='')



.. py:class:: SpearmanMetric(*targets, prefix=None)

   Bases: :class:`kiwi.metrics.metrics.SentenceMetric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. attribute:: _name
      :annotation: = SPEARMAN

      

   .. method:: compute(self, steps, prefix='')



.. py:class:: RMSEMetric(*targets, prefix=None)

   Bases: :class:`kiwi.metrics.metrics.SentenceMetric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. attribute:: _name
      :annotation: = RMSE

      

   .. attribute:: best_ordering
      :annotation: = min

      

   .. method:: compute(self, steps, prefix='')



.. py:class:: PerplexityMetric(*targets, prefix=None)

   Bases: :class:`kiwi.metrics.metrics.Metric`

   .. attribute:: _name
      :annotation: = PERP

      

   .. attribute:: best_ordering
      :annotation: = min

      

   .. method:: step(self, model_out, batch, losses)


   .. method:: compute(self, steps, prefix='')



.. py:class:: ExpectedErrorMetric(*args, labels=None, **kwargs)

   Bases: :class:`kiwi.metrics.metrics.LabeledMetric`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. attribute:: _name
      :annotation: = ExpErr

      

   .. attribute:: best_ordering
      :annotation: = min

      

   .. method:: step(self, model_out, batch, losses)


   .. method:: compute(self, steps, prefix='')



