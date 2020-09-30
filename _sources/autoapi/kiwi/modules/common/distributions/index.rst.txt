:mod:`kiwi.modules.common.distributions`
========================================

.. py:module:: kiwi.modules.common.distributions


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.modules.common.distributions.TruncatedNormal



.. py:class:: TruncatedNormal(loc, scale, lower_bound=0.0, upper_bound=1.0, validate_args=None)

   Bases: :class:`torch.distributions.TransformedDistribution`

   Extension of the Distribution class, which applies a sequence of Transforms
   to a base distribution.  Let f be the composition of transforms applied::

       X ~ BaseDistribution
       Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
       log p(Y) = log p(X) + log |det (dX/dY)|

   Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
   maximum shape of its base distribution and its transforms, since transforms
   can introduce correlations among events.

   An example for the usage of :class:`TransformedDistribution` would be::

       # Building a Logistic Distribution
       # X ~ Uniform(0, 1)
       # f = a + b * logit(X)
       # Y ~ f(X) ~ Logistic(a, b)
       base_distribution = Uniform(0, 1)
       transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
       logistic = TransformedDistribution(base_distribution, transforms)

   For more examples, please look at the implementations of
   :class:`~torch.distributions.gumbel.Gumbel`,
   :class:`~torch.distributions.half_cauchy.HalfCauchy`,
   :class:`~torch.distributions.half_normal.HalfNormal`,
   :class:`~torch.distributions.log_normal.LogNormal`,
   :class:`~torch.distributions.pareto.Pareto`,
   :class:`~torch.distributions.weibull.Weibull`,
   :class:`~torch.distributions.relaxed_bernoulli.RelaxedBernoulli` and
   :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`

   .. attribute:: arg_constraints
      

      

   .. attribute:: support
      

      

   .. attribute:: has_rsample
      :annotation: = True

      

   .. method:: partition_function(self)


   .. method:: scale(self)
      :property:


   .. method:: mean(self)
      :property:

      :math:`pdf = f(x; \mu, \sigma, a, b) = \frac{\phi(\xi)}{\sigma Z}`

      :math:`\xi=\frac{x-\mu}{\sigma}`

      :math:`\alpha=\frac{a-\mu}{\sigma}`

      :math:`\beta=\frac{b-\mu}{\sigma}`

      :math:`Z=\Phi(\beta)-\Phi(\alpha)`

      :returns: :math:`\mu +  \frac{\phi(\alpha)-\phi(\beta)}{Z}\sigma`


   .. method:: variance(self)
      :property:

      Returns the variance of the distribution.


   .. method:: log_prob(self, value)

      Scores the sample by inverting the transform(s) and computing the score
      using the score of the base distribution and the log abs det jacobian.


   .. method:: cdf(self, value)

      Computes the cumulative distribution function by inverting the
      transform(s) and computing the score of the base distribution.


   .. method:: icdf(self, value)

      Computes the inverse cumulative distribution function using
      transform(s) and computing the score of the base distribution.



