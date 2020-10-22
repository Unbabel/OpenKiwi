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
from torch import nn as nn

from kiwi.modules.common.distributions import TruncatedNormal
from kiwi.modules.common.feedforward import feedforward
from kiwi.utils.tensors import sequence_mask


class SentenceFromLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_out = nn.Linear(2, 1)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0.0)

        self._loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, inputs, lengths):
        mask = sequence_mask(lengths, max_len=inputs.size(1)).float()
        average = (inputs * mask[..., None]).sum(1) / lengths[:, None].float()
        h = self.linear_out(average)
        return h

    def loss_fn(self, predicted, target):
        target = target[..., None]
        return self._loss_fn(predicted, target)


class SentenceScoreRegression(nn.Module):
    def __init__(
        self,
        input_size,
        dropout=0.0,
        activation=nn.Tanh,
        final_activation=False,
        num_layers=3,
    ):
        super().__init__()

        self.sentence_pred = feedforward(
            input_size,
            n_layers=num_layers,
            out_dim=1,
            dropout=dropout,
            activation=activation,
            final_activation=final_activation,
        )

        self.loss_fn = nn.MSELoss(reduction='sum')

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, batch_inputs):
        sentence_scores = self.sentence_pred(features).squeeze()
        return sentence_scores


class SentenceScoreDistribution(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.sentence_pred = feedforward(input_size, n_layers=3, out_dim=1)
        # Predict truncated Gaussian distribution
        self.sentence_sigma = feedforward(
            input_size,
            n_layers=3,
            out_dim=1,
            activation=nn.Sigmoid,
            final_activation=True,
        )

        self.loss_fn = self._loss_fn

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _loss_fn(predicted, target):
        _, (mu, sigma) = predicted
        # Compute log-likelihood of x given mu, sigma
        dist = TruncatedNormal(mu, sigma + 1e-12, 0.0, 1.0)
        nll = -dist.log_prob(target)
        return nll.sum()

    def forward(self, features, batch_inputs):
        mu = self.sentence_pred(features).squeeze()
        sigma = self.sentence_sigma(features).squeeze()
        # Compute the mean of the truncated Gaussian as sentence score prediction
        dist = TruncatedNormal(
            mu.clone().detach(), sigma.clone().detach() + 1e-12, 0.0, 1.0
        )
        sentence_scores = dist.mean
        return sentence_scores.squeeze(), (mu, sigma)


class BinarySentenceScore(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.sentence_pred = feedforward(
            input_size, n_layers=3, out_dim=2, activation=nn.Tanh
        )

        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, batch_inputs):
        return self.sentence_pred(features).squeeze()
