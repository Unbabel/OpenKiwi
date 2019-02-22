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

import math
import time
from collections import OrderedDict

import numpy as np
import torch
from scipy.stats.stats import pearsonr, spearmanr
from torch import nn

from kiwi import constants as const
from kiwi.metrics.functions import fscore, precision_recall_fscore_support
from kiwi.models.utils import replace_token


class Metric:
    def __init__(
        self,
        target_name=None,
        metric_name=None,
        PAD=None,
        STOP=None,
        prefix=None,
    ):
        super().__init__()
        self.reset()
        self.prefix = prefix
        self.target_name = target_name
        self.metric_name = metric_name
        self.PAD = PAD
        self.STOP = STOP

    def update(self, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def summarize(self, **kwargs):
        raise NotImplementedError

    def get_name(self):
        return self._prefix(self.metric_name)

    def _prefix_keys(self, summary):
        if self.prefix:
            summary = OrderedDict(
                {self._prefix(key): value for key, value in summary.items()}
            )
        return summary

    def _prefix(self, key):
        if self.prefix:
            return '{}_{}'.format(self.prefix, key)
        return key

    def token_mask(self, batch):
        target = self.get_target(batch)
        if self.PAD is not None:
            return target != self.PAD
        else:
            return torch.ones(
                target.shape, dtype=torch.uint8, device=target.device
            )

    def get_target(self, batch):
        target = getattr(batch, self.target_name)
        if self.STOP is not None:
            target = replace_token(target[:, 1:-1], self.STOP, self.PAD)
        return target

    def get_token_indices(self, batch):
        mask = self.token_mask(batch)
        return mask.view(-1).nonzero().squeeze()

    def get_predictions(self, model_out):
        predictions = model_out[self.target_name]
        return predictions

    def get_target_flat(self, batch):
        target_flat = self.get_target(batch).contiguous().view(-1)
        token_indices = self.get_token_indices(batch)
        return target_flat[token_indices]

    def get_predictions_flat(self, model_out, batch):
        predictions = self.get_predictions(model_out).contiguous()
        predictions_flat = predictions.view(-1, predictions.shape[-1]).squeeze()
        token_indices = self.get_token_indices(batch)
        return predictions_flat[token_indices]

    def get_tokens(self, batch):
        return self.token_mask(batch).sum().item()


class NLLMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='NLL', **kwargs)

    def update(self, loss, batch, **kwargs):
        self.tokens += self.get_tokens(batch)
        self.nll += loss[self.target_name].item()

    def summarize(self):
        summary = {self.metric_name: self.nll / self.tokens}
        return self._prefix_keys(summary)

    def reset(self):
        self.nll = 0.0
        self.tokens = 0


class PerplexityMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='PERP', **kwargs)

    def reset(self):
        self.tokens = 0
        self.nll = 0.0

    def update(self, loss, batch, **kwargs):
        self.tokens += self.get_tokens(batch)
        self.nll += loss[self.target_name].item()

    def summarize(self):
        summary = {self.metric_name: math.e ** (self.nll / self.tokens)}
        return self._prefix_keys(summary)


class CorrectMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='CORRECT', **kwargs)

    def update(self, model_out, batch, **kwargs):
        self.tokens += self.get_tokens(batch)
        logits = self.get_predictions_flat(model_out, batch)
        target = self.get_target_flat(batch)
        _, pred = logits.max(-1)
        correct = target == pred
        correct_count = correct.sum().item()
        self.correct += correct_count

    def summarize(self):
        summary = {self.metric_name: float(self.correct) / self.tokens}
        return self._prefix_keys(summary)

    def reset(self):
        self.correct = 0
        self.tokens = 0


class F1Metric(Metric):
    def __init__(self, labels, **kwargs):
        super().__init__(metric_name='F1_MULT', **kwargs)
        self.labels = labels

    def update(self, model_out, batch, **kwargs):
        logits = self.get_predictions_flat(model_out, batch)
        target = self.get_target_flat(batch)
        _, y_hat = logits.max(-1)
        self.Y_HAT += y_hat.tolist()
        self.Y += target.tolist()

    def summarize(self):
        summary = OrderedDict()
        _, _, f1, _ = precision_recall_fscore_support(self.Y_HAT, self.Y)
        summary[self.metric_name] = np.prod(f1)
        for i, label in enumerate(self.labels):
            summary['F1_' + label] = f1[i]
        return self._prefix_keys(summary)

    def reset(self):
        self.Y = []
        self.Y_HAT = []


class PearsonMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='PEARSON', **kwargs)

    def reset(self):
        self.predictions = []
        self.target = []

    def update(self, model_out, batch, **kwargs):
        target = self.get_target_flat(batch)
        predictions = self.get_predictions_flat(model_out, batch)
        self.predictions += predictions.tolist()
        self.target += target.tolist()

    def summarize(self):
        pearson = pearsonr(self.predictions, self.target)[0]
        summary = {self.metric_name: pearson}
        return self._prefix_keys(summary)


class SpearmanMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='SPEARMAN', **kwargs)

    def reset(self):
        self.predictions = []
        self.target = []

    def update(self, model_out, batch, **kwargs):
        target = self.get_target_flat(batch)
        predictions = self.get_predictions_flat(model_out, batch)
        self.predictions += predictions.tolist()
        self.target += target.tolist()

    def summarize(self):
        spearman = spearmanr(self.predictions, self.target)[0]
        summary = {self.metric_name: spearman}
        return self._prefix_keys(summary)


class ExpectedErrorMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='ExpErr', **kwargs)

    def update(self, model_out, batch, **kwargs):
        logits = self.get_predictions_flat(model_out, batch)
        target = self.get_target_flat(batch)
        probs = nn.functional.softmax(logits, -1)
        probs = probs.gather(-1, target.unsqueeze(-1)).squeeze()
        errors = 1.0 - probs
        self.tokens += self.get_tokens(batch)
        self.expected_error += errors.sum().item()

    def summarize(self):
        summary = {self.metric_name: self.expected_error / self.tokens}
        return self._prefix_keys(summary)

    def reset(self):
        self.expected_error = 0.0
        self.tokens = 0


class TokPerSecMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='TokPerSec', **kwargs)

    def update(self, batch, **kwargs):
        self.tokens += self.get_tokens(batch)

    def summarize(self):
        summary = {self.metric_name: self.tokens / (time.time() - self.time)}
        return self._prefix_keys(summary)

    def reset(self):
        self.tokens = 0
        self.time = time.time()


class LogMetric(Metric):
    """Logs averages of values in loss, model or batch.
    """

    def __init__(self, targets, metric_name=None, **kwargs):
        self.targets = targets
        metric_name = metric_name or self._format(*targets[0])
        super().__init__(metric_name=metric_name, **kwargs)

    def update(self, **kwargs):
        self.steps += 1
        for side, target in self.targets:
            key = self._format(side, target)
            self.log[key] += kwargs[side][target].mean().item()

    def summarize(self):
        summary = {
            key: value / float(self.steps) for key, value in self.log.items()
        }
        return self._prefix_keys(summary)

    def reset(self):
        self.log = {
            self._format(side, target): 0.0 for side, target in self.targets
        }
        self.steps = 0

    def _format(self, side, target):
        return '{}_{}'.format(side, target)


class RMSEMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='RMSE', **kwargs)

    def update(self, batch, model_out, **kwargs):
        predictions = self.get_predictions_flat(model_out, batch)
        target = self.get_target_flat(batch)
        self.squared_error += ((predictions - target) ** 2).sum().item()
        self.tokens += self.get_tokens(batch)

    def summarize(self):
        rmse = math.sqrt(self.squared_error / self.tokens)
        summary = {self.metric_name: rmse}
        return self._prefix_keys(summary)

    def reset(self):
        self.squared_error = 0.0
        self.tokens = 0


class TokenMetric(Metric):
    def __init__(self, target_token=const.UNK_ID, token_name='UNK', **kwargs):
        self.target_token = target_token
        super().__init__(metric_name='UNKS', **kwargs)

    def update(self, batch, **kwargs):
        target = self.get_target_flat(batch)
        self.targets += (target == self.target_token).sum().item()
        self.tokens += self.get_tokens(batch)

    def summarize(self):
        summary = {}
        if self.tokens:
            summary = {self.metric_name: self.targets / self.tokens}
        return self._prefix_keys(summary)

    def reset(self):
        self.tokens = 0
        self.targets = 0


class ThresholdCalibrationMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='F1_CAL', **kwargs)

    def update(self, model_out, batch, **kwargs):
        logits = self.get_predictions_flat(model_out, batch)
        bad_probs = nn.functional.softmax(logits, -1)[:, const.BAD_ID]
        target = self.get_target_flat(batch)
        self.scores += bad_probs.tolist()
        self.Y += target.tolist()

    def summarize(self):
        summary = {}
        mid = len(self.Y) // 2
        if mid:
            perm = np.random.permutation(len(self.Y))
            self.Y = [self.Y[idx] for idx in perm]
            self.scores = [self.scores[idx] for idx in perm]
            m = MovingF1()
            fscore, threshold = m.choose(
                m.eval(self.scores[:mid], self.Y[:mid])
            )
            predictions = [
                const.BAD_ID if score >= threshold else const.OK_ID
                for score in self.scores[mid:]
            ]
            _, _, f1, _ = precision_recall_fscore_support(
                predictions, self.Y[mid:]
            )
            f1_mult = np.prod(f1)
            summary = {self.metric_name: f1_mult}
        return self._prefix_keys(summary)

    def reset(self):
        self.scores = []
        self.Y = []


class MovingMetric:
    """Class to compute the changes in one metric as a function of a second metric.
       Example: F1 score vs. Classification Threshold, Quality vs Skips
    """

    def eval(self, scores, labels):
        """Compute the graph metric1 vs metric2
        Args:
           Scores: Model Outputs
           Labels: Corresponding Labels
        """
        self.init(scores, labels)
        scores, labels = self.sort(scores, labels)
        init_threshold = scores[0]
        thresholds = [(self.compute(), init_threshold)]
        for score, label in zip(scores, labels):
            self.update(score, label)
            thresholds.append((self.compute(), score))
        return thresholds

    def init(self, scores, labels):
        """Initialize the Metric for threshold < min(scores)
        """
        return scores, labels

    def sort(self, scores, labels):
        """Sort List of labels and scores.
        """
        return zip(*sorted(zip(scores, labels)))

    def update(self, score, label):
        """Move the threshold past score
        """
        return None

    def compute(self):
        """Compute the current Value of the metric
        """
        pass

    def choose(self, thresholds):
        """Choose the best (threshold, metric) tuple from an iterable.
        """
        pass


class MovingF1(MovingMetric):
    def init(self, scores, labels, class_idx=1):
        """
        Compute F1 Mult for all decision thresholds over (scores, labels)
        Initialize the threshold s.t. all examples are classified as
        `class_idx`.
        Args:
           scores: Likelihood scores for class index
           Labels: Gold Truth classes in {0,1}
           class_index: ID of class
        """
        # -1 if class_idx == 0 , 1 if class_idx == 1
        self.sign = 2 * class_idx - 1
        class_one = sum(labels)
        class_zero = len(labels) - class_one
        self.fp_zero = (1 - class_idx) * class_one
        self.tp_zero = (1 - class_idx) * class_zero
        self.fp_one = class_idx * class_zero
        self.tp_one = class_idx * class_one

    def update(self, score, label):
        """Move the decision threshold.
        """
        self.tp_zero += self.sign * (1 - label)
        self.fp_zero += self.sign * label
        self.tp_one -= self.sign * label
        self.fp_one -= self.sign * (1 - label)

    def compute(self):
        f1_zero = fscore(self.tp_zero, self.fp_zero, self.fp_one)
        f1_one = fscore(self.tp_one, self.fp_one, self.fp_zero)
        return f1_one * f1_zero

    def choose(self, thresholds):
        return max(thresholds)


class MovingSkipsAtQuality(MovingMetric):
    """Computes Quality of skipped examples vs fraction of skips.
    """

    def __init__(
        self, scores_higher_is_better=False, labels_higher_is_better=False
    ):
        """
          Args:
          scores_higher_is_better:
              If True, higher model outputs indicate higher quality.
          labels_higher_is_better:
              If True, higher label values indicate higher quality.
        """
        self.scores_higher_is_better = scores_higher_is_better
        self.labels_higher_is_better = labels_higher_is_better

    def eval(self, scores, labels):
        """
         Args:
          scores: Model output quality or error scores. If quality scores
              are provided, pass scores_higher_is_better=True.
          labels: Ground truth quality or error scores. If quality scores
              are provided, pass labels_higher_is_better=True.
        """
        return super().eval(scores, labels)

    def init(self, scores, labels):
        """
         Args:
          scores: Model output quality or error scores. If quality scores
              are provided, pass scores_higher_is_better=True.
          labels: Ground truth quality or error scores. If quality scores
              are provided, pass labels_higher_is_better=True.
        """
        self.cumulative_qual = 0.0
        self.skipped = 0
        self.data_size = len(scores)

    def update(self, score, label):
        self.cumulative_qual += label
        self.skipped += 1

    def compute(self):
        if not self.skipped:
            return None, 0.0
        return (
            self.skipped / self.data_size,
            self.cumulative_qual / self.skipped,
        )

    def choose(self, thresholds, target_qual):
        """Chooses the smallest threshold such that
           avg. quality  is greater than or equal to target_qual
        """
        best = None
        sign = 1 if self.labels_higher_is_better else -1
        for ((skip, qual), t) in thresholds:
            if (sign * (qual - target_qual)) >= 0:
                # The quality at threshold t is admissible given target_qual
                if best is None:
                    best = ((skip, qual), t)
                else:
                    last_best = abs(best[0][0] - target_qual)
                    if abs(qual - target_qual) < last_best:
                        # The quality at threshold t is admissible given
                        # target_qual and closer than the previous best
                        best = ((skip, qual), t)
        return best

    def sort(self, scores, labels):
        return zip(
            *sorted(zip(scores, labels), reverse=self.scores_higher_is_better)
        )
