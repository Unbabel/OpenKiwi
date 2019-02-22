#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2019 Unbabel <openkiwi@unbabel.com>
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

import functools
import logging
from collections import OrderedDict

from kiwi.loggers import tracking_logger

logger = logging.getLogger(__name__)


@functools.total_ordering
class StatsSummary(OrderedDict):
    def __init__(self, prefix=None, main_metric=None, ordering=max, **kwargs):
        self.prefix = prefix
        self._main_metric_name = main_metric
        self.ordering = ordering
        super().__init__(**kwargs)

    @property
    def main_metric(self):
        if self._main_metric_name:
            return self._main_metric_name
        elif self:
            return list(self.keys())[0]
        return None

    def main_metric_value(self):
        return self.__getitem__(self.main_metric)

    def _make_key(self, key):
        if self.prefix:
            key = '{}_{}'.format(self.prefix, key)
        return key

    def __str__(self):
        return ', '.join(['{}: {:0.4f}'.format(k, v) for k, v in self.items()])

    def log(self):
        """Log statistics to output and also to tracking logger.

        :param stats_summary: StatsSummary object
        """
        print('\r', end='\r')
        logger.info(self)

        for k, v in self.items():
            tracking_logger.log_metric(k, v)

    def __setitem__(self, key, value):
        key = self._make_key(key)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        key = self._make_key(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        key = self._make_key(key)
        return super().__contains__(key)

    def get(self, key, default=None):
        key = self._make_key(key)
        return super().get(key, default)

    def __eq__(self, other):
        return isinstance(other, StatsSummary) and self.get(
            self.main_metric
        ) == other.get(self.main_metric)

    def __le__(self, other):
        if self.ordering == max:
            return isinstance(other, StatsSummary) and self.get(
                self.main_metric
            ) <= other.get(self.main_metric)
        else:
            return isinstance(other, StatsSummary) and self.get(
                self.main_metric
            ) >= other.get(self.main_metric)

    def __gt__(self, other):
        if self.ordering == max:
            return isinstance(other, StatsSummary) and self.get(
                self.main_metric
            ) > other.get(self.main_metric)
        else:
            return isinstance(other, StatsSummary) and self.get(
                self.main_metric
            ) < other.get(self.main_metric)

    def better_than(self, other):
        if self.ordering == max:
            return isinstance(other, StatsSummary) and self.get(
                self.main_metric
            ) > other.get(self.main_metric)
        else:
            return isinstance(other, StatsSummary) and self.get(
                self.main_metric
            ) < other.get(self.main_metric)


class Stats:
    def __init__(
        self,
        metrics,
        main_metric=None,
        main_metric_ordering=max,
        log_interval=0,
    ):
        self.metrics = metrics
        main_metric = main_metric or self.metrics[0]
        self.main_metric_name = main_metric.get_name()
        self.main_metric_ordering = main_metric_ordering

        self.log_interval = log_interval

        self.reset()

    def update(self, **kwargs):
        self.steps += 1
        for metric in self.metrics:
            metric.update(**kwargs)

    def summarize(self, prefix=None):
        summary = StatsSummary(
            prefix=prefix,
            main_metric=self.main_metric_name,
            ordering=self.main_metric_ordering,
        )
        if self.steps:
            for metric in self.metrics:
                summary.update(metric.summarize())
        return summary

    def reset(self):
        self.steps = 0
        for metric in self.metrics:
            metric.reset()

    def wrap_up(self, prefix=None):
        summary = self.summarize(prefix)
        self.reset()
        return summary

    def log(self, step=None):
        if (
            step is None
            or self.log_interval > 0
            and not step % self.log_interval
        ):
            stats_summary = self.wrap_up()
            stats_summary.log()
