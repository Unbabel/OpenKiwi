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

import numpy as np


class StructuredDecoder(object):
    """An abstract decoder for structured prediction."""

    def __init__(self):
        pass

    def decode(self, instance, parts, scores):
        """Decode, computing the highest-scores output.
        Must return a vector of 0/1 predicted_outputs of the same size
        as parts."""
        raise NotImplementedError

    def decode_mira(
        self, instance, parts, scores, gold_outputs, old_mira=False
    ):
        """Perform cost-augmented decoding or classical MIRA."""
        p = 0.5 - gold_outputs
        q = 0.5 * np.ones(len(gold_outputs)).dot(gold_outputs)
        if old_mira:
            predicted_outputs = self.decode(instance, parts, scores)
        else:
            scores_cost = scores + p
            predicted_outputs = self.decode(instance, parts, scores_cost)
        cost = p.dot(predicted_outputs) + q
        loss = cost + scores.dot(predicted_outputs - gold_outputs)

        return predicted_outputs, cost, loss

    def decode_cost_augmented(self, instance, parts, scores, gold_outputs):
        """Perform cost-augmented decoding."""
        return self.decode_mira(
            instance, parts, scores, gold_outputs, old_mira=False
        )
