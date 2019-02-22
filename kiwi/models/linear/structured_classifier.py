"""A generic implementation of an abstract structured linear classifier."""

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

import logging

import numpy as np

from .linear_model import LinearModel
from .structured_decoder import StructuredDecoder
from .utils import nearly_eq_tol

logger = logging.getLogger(__name__)


class StructuredClassifier:
    """ An abstract structured classifier."""

    def __init__(self):
        self.model = LinearModel()
        self.decoder = StructuredDecoder()
        self.use_binary_features = False
        self.feature_indices = None

    def save(self, model_path):
        """Save the full configuration and model."""
        raise NotImplementedError

    def load(self, model_path):
        """Load the full configuration and model."""
        raise NotImplementedError

    def create_instances(self, dataset):
        """Preprocess the dataset if needed to create instances.
        Default is returning the dataset itself. Override if needed."""
        return dataset

    def label_instance(self, instance, parts, predicted_output):
        """Return a labeled instance by adding the predicted output
        information."""
        raise NotImplementedError

    def create_prediction(self, instance, parts, predicted_output):
        """Create a prediction for an instance."""
        raise NotImplementedError

    def make_parts(self, instance):
        """Compute the task-specific parts for this instance."""
        raise NotImplementedError

    def make_features(self, instance, parts):
        """Create a feature vector for each part."""
        raise NotImplementedError

    def compute_scores(self, instance, parts, features):
        """Compute a score for every part in the instance using the current
        model and the part-specific features."""
        num_parts = len(parts)
        scores = np.zeros(num_parts)
        for r in range(num_parts):
            if self.use_binary_features:
                scores[r] = self.model.compute_score_binary_features(
                    features[r]
                )
            else:
                scores[r] = self.model.compute_score(features[r])
        return scores

    def run(self, instance):
        """Run the structured classifier on a single instance."""
        parts, gold_output = self.make_parts(instance)
        features = self.make_features(instance, parts)
        scores = self.compute_scores(instance, parts, features)
        predicted_output = self.decoder.decode(instance, parts, scores)
        labeled_instance = self.label_instance(
            instance, parts, predicted_output
        )
        return labeled_instance

    def test(self, instances):
        """Run the structured classifier on dev/test data."""
        num_mistakes = 0
        num_parts_total = 0
        predictions = []
        for instance in instances:
            # TODO: use self.run(instance) instead?
            parts, gold_output = self.make_parts(instance)
            features = self.make_features(instance, parts)
            scores = self.compute_scores(instance, parts, features)
            predicted_output = self.decoder.decode(instance, parts, scores)
            predictions.append(
                self.create_prediction(instance, parts, predicted_output)
            )
            num_parts = len(parts)
            assert len(predicted_output) == num_parts
            assert len(gold_output) == num_parts
            for i in range(num_parts):
                if not nearly_eq_tol(gold_output[i], predicted_output[i], 1e-6):
                    num_mistakes += 1
            num_parts_total += num_parts

        logger.info(
            'Part accuracy: %f',
            float(num_parts_total - num_mistakes) / float(num_parts_total),
        )
        return predictions

    def evaluate(self, instances, predictions, print_scores=True):
        """Evaluate the structure classifier, computing a task-dependent
        evaluation metric."""
        raise NotImplementedError
