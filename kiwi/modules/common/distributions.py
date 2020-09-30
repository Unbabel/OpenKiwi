#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2020 Unbabel <openkiwi@unbabel.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from torch.distributions import (
    Normal,
    TransformedDistribution,
    constraints,
    identity_transform,
)


class TruncatedNormal(TransformedDistribution):
    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'lower_bound': constraints.real,
        'upper_bound': constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self, loc, scale, lower_bound=0.0, upper_bound=1.0, validate_args=None
    ):
        base_dist = Normal(loc, scale)
        super(TruncatedNormal, self).__init__(
            base_dist, identity_transform, validate_args=validate_args
        )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def partition_function(self):
        return (
            self.base_dist.cdf(self.upper_bound) - self.base_dist.cdf(self.lower_bound)
        ).detach() + 1e-12

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        r"""
        :math:`pdf = f(x; \mu, \sigma, a, b) = \frac{\phi(\xi)}{\sigma Z}`

        :math:`\xi=\frac{x-\mu}{\sigma}`

        :math:`\alpha=\frac{a-\mu}{\sigma}`

        :math:`\beta=\frac{b-\mu}{\sigma}`

        :math:`Z=\Phi(\beta)-\Phi(\alpha)`

        Return:
            :math:`\mu +  \frac{\phi(\alpha)-\phi(\beta)}{Z}\sigma`

        """
        mean = self.base_dist.mean + (
            (
                self.base_dist.scale ** 2
                * (
                    self.base_dist.log_prob(self.lower_bound).exp()
                    - self.base_dist.log_prob(self.upper_bound).exp()
                )
            )
            / self.partition_function()
        )
        return mean

    @property
    def variance(self):
        pdf_a = self.base_dist.log_prob(self.lower_bound).exp()
        pdf_b = self.base_dist.log_prob(self.upper_bound).exp()
        alpha = (
            self.lower_bound - self.base_dist.mean
        ) * self.base_dist.scale.reciprocal()
        beta = (
            self.upper_bound - self.base_dist.mean
        ) * self.base_dist.scale.reciprocal()
        z = self.partition_function()
        term1 = (alpha * pdf_a - beta * pdf_b) / z
        term2 = (pdf_a - pdf_b) / z
        return self.base_dist.scale ** 2 * (1 + term1 - term2 ** 2)

    def log_prob(self, value):
        log_value = self.base_dist.log_prob(value)
        log_prob = log_value - self.partition_function().log()
        return log_prob

    def cdf(self, value):
        if value <= self.lower_bound:
            return 0.0
        if value >= self.upper_bound:
            return 1.0
        unnormalized_cdf = self.base_dist.cdf(value) - self.base_dist.cdf(
            self.lower_bound
        )
        return unnormalized_cdf / self.partition_function()

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.base_dist.icdf(
            self.cdf(self.lower_bound) + value * self.partition_function()
        )
