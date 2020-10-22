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
