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
import math
from abc import ABC
from collections import OrderedDict

import numpy as np
import torch
from scipy.stats.stats import pearsonr, spearmanr
from torch import nn

from kiwi.metrics.functions import (
    fscore,
    matthews_correlation_coefficient,
    precision_recall_fscore_support,
)


class Metric:
    _name = None
    best_ordering = 'max'

    def __init__(self, *targets, prefix=None):
        self.targets = targets

        if prefix is None:
            prefix = f'{self.targets[0]}_'
        self.prefix = prefix

    def step(self, model_out, batch, losses):
        raise NotImplementedError

    def compute(self, steps, prefix=''):
        raise NotImplementedError

    @property
    def name(self):
        return f'{self.prefix}{self._name}'

    def num_tokens(self, batch, *targets):
        if not targets:
            targets = self.targets
        return sum([batch[target].strict_masks.sum().item() for target in targets])


class NLLMetric(Metric):
    _name = 'NLL'
    best_ordering = 'min'

    def step(self, model_out, batch, losses):
        nll = sum(losses[target].item() for target in self.targets)
        nr_tokens = self.num_tokens(batch)
        return nll, nr_tokens

    def compute(self, steps, prefix=''):
        if not steps:
            return {}
        nll, nr_tokens = zip(*steps)
        nll = torch.cat(nll).sum()
        nr_tokens = torch.cat(nr_tokens).sum()

        summary = {f'{prefix}{self.name}': nll / nr_tokens}
        return summary


class LabeledMetric(Metric, ABC):
    def __init__(self, *args, labels=None, **kwargs):
        assert labels is not None
        super().__init__(*args, **kwargs)
        self.labels = labels

    def step(self, model_out, batch, losses):
        logits = self.get_predictions_flat(model_out, batch)
        _, y_hat = logits.max(-1)
        target = self.get_target_flat(batch)
        return y_hat, target

    def get_target_flat(self, batch):
        targets = [batch[target].tensor for target in self.targets]
        masks = [batch[target].strict_masks for target in self.targets]
        targets = torch.cat(targets, dim=1)
        masks = torch.cat(masks, dim=1)
        return targets[masks]  # this flattens out the tensor

    def get_predictions_flat(self, model_out, batch):
        predictions = [model_out[target] for target in self.targets]
        masks = [batch[target].strict_masks for target in self.targets]
        predictions = torch.cat(predictions, dim=1)
        masks = torch.cat(masks, dim=1)
        # In QESystems, predictions don't contain BOS/EOS, neither their batch fields.
        #  We rely on this consistent behaviour, i.e., the model will output according
        #  to the batched target (either both have BOS/EOS, or neither does).
        return predictions[masks]  # this flattens out the tensor


# Word-level metrics


class CorrectMetric(LabeledMetric):
    _name = 'CORRECT'

    def step(self, model_out, batch, losses):
        predictions = model_out[self.targets[0]]
        if len(predictions.shape) > 2:
            logits = self.get_predictions_flat(model_out, batch)
            _, y_hat = logits.max(-1)
            target = self.get_target_flat(batch)
        else:
            _, y_hat = predictions.max(-1)
            target = batch[self.targets[0]]
        return y_hat, target

    def compute(self, steps, prefix=''):
        if not steps:
            return {}
        y_hat, y = zip(*steps)
        y_hat = torch.cat(y_hat)
        y = torch.cat(y)

        correct = y_hat == y
        correct_count = correct.sum(dtype=torch.float)
        summary = {f'{prefix}{self.name}': correct_count / len(y)}
        return summary


class F1MultMetric(LabeledMetric):
    _name = 'F1_MULT'

    def compute(self, steps, prefix=''):
        if not steps:
            return {}
        y_hat, y = zip(*steps)
        y_hat = torch.cat(y_hat)
        y = torch.cat(y)

        _, _, f1, _ = precision_recall_fscore_support(y_hat, y, labels=self.labels)
        summary = OrderedDict()
        summary[f'{prefix}{self.name}'] = np.prod(f1)
        for i, label in enumerate(self.labels):
            summary[f'{prefix}F1_{label}'] = torch.tensor(f1[i])
        return summary


class MatthewsMetric(LabeledMetric):
    _name = 'MCC'

    def compute(self, steps, prefix=''):
        if not steps:
            return {}
        y_hat, y = zip(*steps)
        y_hat = torch.cat(y_hat)
        y = torch.cat(y)

        matthews = matthews_correlation_coefficient(y_hat, y)
        summary = {f'{prefix}{self.name}': torch.tensor(matthews)}
        return summary


# Sentence-level metrics


class SentenceMetric(Metric, ABC):
    def step(self, model_out, batch, losses):
        y_hat = []
        for target in self.targets:
            out = model_out[target]
            if isinstance(out, tuple):
                # Output that returns extra info; by convention, first element must be
                #  the score.
                out = out[0]
            y_hat.append(out.detach().cpu())
        y_hat = torch.cat(y_hat)
        y = torch.cat([batch[target].detach().cpu() for target in self.targets])
        return y_hat, y


class PearsonMetric(SentenceMetric):
    _name = 'PEARSON'

    def compute(self, steps, prefix=''):
        if not steps:
            return {}
        y_hat, y = zip(*steps)
        y_hat = torch.cat(y_hat)
        y = torch.cat(y)

        pearson, _ = pearsonr(y_hat, y)
        summary = {f'{prefix}{self.name}': torch.tensor(pearson)}
        return summary


class SpearmanMetric(SentenceMetric):
    _name = 'SPEARMAN'

    def compute(self, steps, prefix=''):
        if not steps:
            return {}
        y_hat, y = zip(*steps)
        y_hat = torch.cat(y_hat)
        y = torch.cat(y)

        spearman, *_ = spearmanr(y_hat.numpy(), y.numpy())
        summary = {f'{prefix}{self.name}': torch.tensor(spearman)}
        return summary


class RMSEMetric(SentenceMetric):
    _name = 'RMSE'
    best_ordering = 'min'

    def compute(self, steps, prefix=''):
        if not steps:
            return {}
        y_hat, y = zip(*steps)
        y_hat = torch.cat(y_hat)
        y = torch.cat(y)

        rmse = (((y_hat - y) ** 2).sum() / len(y)) ** (1 / 2)
        summary = {f'{prefix}{self.name}': rmse}
        return summary


# TLM metrics


class PerplexityMetric(Metric):
    _name = 'PERP'
    best_ordering = 'min'

    def step(self, model_out, batch, losses):
        # TODO: is this the right way of calculating perplexity?
        nll = sum(losses[target].item() for target in self.targets)
        nr_tokens = self.num_tokens(batch, 'target')
        return nll, nr_tokens

    def compute(self, steps, prefix=''):
        if not steps:
            return {}
        nll, nr_tokens = zip(*steps)
        nll = sum(nll)
        nr_tokens = sum(nr_tokens)

        perplexity = math.e ** (nll / nr_tokens)
        summary = {f'{prefix}{self.name}': perplexity}
        return summary


class ExpectedErrorMetric(LabeledMetric):
    _name = 'ExpErr'
    best_ordering = 'min'

    def step(self, model_out, batch, losses):
        logits = self.get_predictions_flat(model_out, batch)
        target = self.get_target_flat(batch)
        probs = nn.functional.softmax(logits, -1)
        probs = probs.gather(-1, target.unsqueeze(-1)).squeeze()
        errors = 1.0 - probs
        tokens = self.num_tokens(batch)
        expected_error = errors.sum().item()
        return expected_error, tokens

    def compute(self, steps, prefix=''):
        if not steps:
            return {}
        e_err, nr_tokens = zip(*steps)
        e_err = sum(e_err)
        nr_tokens = sum(nr_tokens)

        expected_error = e_err / nr_tokens
        summary = {f'{prefix}{self.name}': expected_error}
        return summary


# Other metrics


# class TokPerSecMetric(Metric):
#     _name = 'TokPerSec'
#     best_ordering = 'min'
#
#     def step(self, model_out, batch, losses):
#         nr_tokens = self.num_tokens(batch, 'target')
#         timestamp = time.time()
#         return nr_tokens, timestamp
#
#     def compute(self, steps, prefix=''):
#         if not steps:
#             return {}
#         nr_tokens, timestamps = zip(*steps)
#         nr_tokens = torch.cat(nr_tokens).sum().item()
#         delta_time = (time.time() - timestamps[0])
#         summary = {f'{prefix}{self.name}': nr_tokens / delta_time}
#         return summary


# class LogMetric(Metric):
#     """Logs averages of values in loss, model or batch.
#     """
#
#     def __init__(self, log_targets, name=None, **kwargs):
#         self.log_targets = log_targets
#         name = name or self._format(*log_targets[0])
#         super().__init__(name=name, **kwargs)
#
#     def update(self, **kwargs):
#         self.steps += 1
#         for side, target in self.log_targets:
#             key = self._format(side, target)
#             self.log[key] += kwargs[side][target].mean().item()
#
#     def summarize(self):
#         summary = {key: value / float(self.steps) for key, value in self.log.items()}
#         return self._prefix_keys(summary)
#
#     def reset(self):
#         self.log = {
#             self._format(side, target): 0.0 for side, target in self.log_targets
#         }
#         self.steps = 0
#
#     def _format(self, side, target):
#         return '{}_{}'.format(side, target)


class ThresholdCalibrationMetric(Metric):
    def __init__(self, target_id=0, **kwargs):
        self.target_id = target_id
        super().__init__(name='F1_CAL', **kwargs)

    def update(self, model_out, batch, **kwargs):
        logits = self.get_predictions_flat(model_out, batch)
        probs = nn.functional.softmax(logits, -1)[:, self.target_id]
        target = self.get_target_flat(batch)
        self.probs += probs.tolist()
        self.Y += target.tolist()

    def summarize(self):
        summary = {}
        mid = len(self.Y) // 2
        if mid:
            perm = np.random.permutation(len(self.Y))
            self.Y = [self.Y[idx] for idx in perm]
            self.probs = [self.probs[idx] for idx in perm]
            m = MovingF1()
            fscore, threshold = m.choose(m.eval(self.probs[:mid], self.Y[:mid]))

            predictions = []
            for prob in self.probs[mid:]:
                if prob >= threshold:
                    predictions.append(self.target_id)
                else:
                    predictions.append(max(0, 1 - self.target_id))
            _, _, f1, _ = precision_recall_fscore_support(predictions, self.Y[mid:])
            f1_mult = np.prod(f1)
            summary = {self.name: f1_mult}
        return self._prefix_keys(summary)

    def reset(self):
        self.probs = []
        self.Y = []


class MovingMetric:
    """Class to compute the changes in one metric as a function of a second metric.

       Example: F1 score vs. Classification Threshold, Quality vs Skips
    """

    def eval(self, scores, labels):
        """Compute the graph metric1 vs metric2.

        Arguments:
           scores: model outputs.
           labels: corresponding labels.
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
        """Initialize the Metric for threshold < min(scores)."""
        return scores, labels

    def sort(self, scores, labels):
        """Sort List of labels and scores."""
        return zip(*sorted(zip(scores, labels)))

    def update(self, score, label):
        """Move the threshold past score."""
        return None

    def compute(self):
        """Compute the current value of the metric."""
        pass

    def choose(self, thresholds):
        """Choose the best (threshold, metric) tuple from an iterable."""
        pass


class MovingF1(MovingMetric):
    def init(self, scores, labels, class_idx=1):
        """Compute F1-Mult for all decision thresholds over (scores, labels).

        Initialize the threshold s.t. all examples are classified as `class_idx`.

        Arguments:
           scores: likelihood scores for class index.
           labels: gold truth classes in {0,1}.
           class_idx: ID of class.
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
        """Move the decision threshold."""
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
    def __init__(self, scores_higher_is_better=False, labels_higher_is_better=False):
        """Compute Quality of skipped examples vs fraction of skips.

        Arguments:
            scores_higher_is_better: whether higher model outputs indicate higher
                                     quality.
            labels_higher_is_better: whether higher label values indicate higher
                                     quality.
        """
        self.scores_higher_is_better = scores_higher_is_better
        self.labels_higher_is_better = labels_higher_is_better

    def eval(self, scores, labels):
        """
        Arguments:
            scores: model output quality or error scores. If quality scores
                    are provided, pass ``scores_higher_is_better=True``.
            labels: ground truth quality or error scores. If quality scores
                    are provided, pass ``labels_higher_is_better=True``.
        """
        return super().eval(scores, labels)

    def init(self, scores, labels):
        """
        Arguments:
            scores: model output quality or error scores. If quality scores are
                    provided, pass ``scores_higher_is_better=True``.
            labels: ground truth quality or error scores. If quality scores are
                    provided, pass ``labels_higher_is_better=True``.
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
        return (self.skipped / self.data_size, self.cumulative_qual / self.skipped)

    def choose(self, thresholds, target_qual):
        """Choose the smallest threshold such that average quality is greater than or
        equal to target_qual.
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
        return zip(*sorted(zip(scores, labels), reverse=self.scores_higher_is_better))
