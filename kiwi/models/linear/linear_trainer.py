"""A generic implementation of a basic trainer."""

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
from pathlib import Path

import numpy as np

from kiwi import constants as const
from kiwi.models.linear.sparse_vector import SparseVector

from .utils import nearly_eq_tol

logger = logging.getLogger(__name__)


class LinearTrainer(object):
    def __init__(
        self,
        classifier,
        checkpointer,
        algorithm='svm_mira',
        regularization_constant=1e12,
    ):
        self.classifier = classifier
        self.algorithm = algorithm
        self.regularization_constant = regularization_constant
        self.checkpointer = checkpointer
        # Only for training with SGD.
        self.initial_learning_rate = 0.001
        # Only for training with SGD. Change to 'inv' for Pegasos-style
        # updating.
        self.learning_rate_schedule = 'invsqrt'
        # Best metric value (to pick the best iteration).
        self.best_metric_value = -np.inf

    def _make_gradient_step(
        self, parts, features, eta, t, gold_output, predicted_output
    ):
        """Perform a gradient step updating the current model."""
        for r in range(len(parts)):
            if predicted_output[r] == gold_output[r]:
                continue
            if self.classifier.use_binary_features:
                part_features = features[r].to_sparse_vector()
            else:
                part_features = features[r]
            self.classifier.model.make_gradient_step(
                part_features, eta, t, predicted_output[r] - gold_output[r]
            )

    def _make_feature_difference(
        self, parts, features, gold_output, predicted_output
    ):
        """Compute the difference between predicted and gold feature vector."""
        difference = SparseVector()
        for r in range(len(parts)):
            if predicted_output[r] == gold_output[r]:
                continue
            if self.classifier.use_binary_features:
                part_features = features[r].to_sparse_vector()
            else:
                part_features = features[r]
                # FIXME: shouldn't the next line be outside the else?
                difference.add(
                    part_features, predicted_output[r] - gold_output[r]
                )
        return difference

    def run(self, train_iterator, valid_iterator, epochs=50):
        """Train with a general online algorithm."""
        import time

        dataset = self.classifier.create_instances(train_iterator.dataset)
        if not isinstance(valid_iterator, list):
            valid_iterator = [valid_iterator]
        dev_datasets = [
            self.classifier.create_instances(iterator.dataset)
            for iterator in valid_iterator
        ]

        self.classifier.model.clear()
        for epoch in range(epochs):
            tic = time.time()
            logger.info('Epoch %d' % (epoch + 1))
            self._train_epoch(epoch, dataset, dev_datasets)
            toc = time.time()
            logger.info('Elapsed time (epoch): %d' % (toc - tic))
        if self.algorithm != 'svm_sgd':
            self.classifier.model.finalize(len(train_iterator.dataset) * epochs)

        self.checkpointer.check_out()

    def _train_epoch(self, epoch, dataset, dev_datasets):
        """Run one epoch of an online algorithm."""
        algorithm = self.algorithm
        total_loss = 0.0
        total_cost = 0.0
        if algorithm in ['perceptron']:
            num_mistakes = 0
            num_total = 0
        elif algorithm in ['mira', 'svm_mira']:
            truncated = 0
        lambda_coefficient = 1.0 / (
            self.regularization_constant * float(len(dataset))
        )
        t = len(dataset) * epoch

        for instance in dataset:
            # Compute parts, features, and scores.
            parts, gold_output = self.classifier.make_parts(instance)
            features = self.classifier.make_features(instance, parts)
            scores = self.classifier.compute_scores(instance, parts, features)

            # Do the decoding.
            if algorithm in ['perceptron']:
                predicted_output = self.classifier.decoder.decode(
                    instance, parts, scores
                )
                for r in range(len(parts)):
                    num_total += 1
                    if not nearly_eq_tol(
                        gold_output[r], predicted_output[r], 1e-6
                    ):
                        num_mistakes += 1

            elif algorithm in ['mira']:
                predicted_output, cost, loss = self.classifier.decoder.decode_mira(  # NOQA
                    instance, parts, scores, gold_output, True
                )

            elif algorithm in ['svm_mira', 'svm_sgd']:
                predicted_output, cost, loss = self.classifier.decoder.decode_cost_augmented(  # NOQA
                    instance, parts, scores, gold_output
                )
            else:
                raise NotImplementedError

            # Update the total loss and cost.
            if algorithm in ['mira', 'svm_mira', 'svm_sgd']:
                if loss < 0.0:
                    if loss < -1e-12:
                        logger.warning('Negative loss: ' + str(loss))
                    loss = 0.0
                if cost < 0.0:
                    if cost < -1e-12:
                        logger.warning('Negative cost:' + str(cost))
                    cost = 0.0

                total_loss += loss
                total_cost += cost

            num_parts = len(parts)
            assert len(gold_output) == num_parts
            assert len(predicted_output) == num_parts

            # Compute the stepsize.
            if algorithm in ['perceptron']:
                eta = 1.0
            elif algorithm in ['mira', 'svm_mira']:
                difference = self._make_feature_difference(
                    parts, features, gold_output, predicted_output
                )
                squared_norm = difference.squared_norm()
                threshold = 1e-9
                if loss < threshold or squared_norm < threshold:
                    eta = 0.0
                else:
                    eta = loss / squared_norm
                if eta > self.regularization_constant:
                    eta = self.regularization_constant
                    truncated += 1
            elif algorithm in ['svm_sgd']:
                if self.learning_rate_schedule == 'invsqrt':
                    eta = self.initial_learning_rate / np.sqrt(float(t + 1))
                elif self.learning_rate_schedule == 'inv':
                    eta = self.initial_learning_rate / (float(t + 1))
                else:
                    raise NotImplementedError

                # Scale the weight vector.
                decay = 1.0 - eta * lambda_coefficient
                assert decay >= -1e-12
                self.classifier.model.weights.scale(decay)

            # Make gradient step.
            self._make_gradient_step(
                parts, features, eta, t, gold_output, predicted_output
            )

            # Increment the round.
            t += 1

        # Evaluate on development data.
        weights = self.classifier.model.weights.copy()
        averaged_weights = self.classifier.model.averaged_weights.copy()
        if algorithm != 'svm_sgd':
            self.classifier.model.finalize(len(dataset) * (1 + epoch))

        dev_scores = []
        for dev_dataset in dev_datasets:
            predictions = self.classifier.test(dev_dataset)
            dev_score = self.classifier.evaluate(
                dev_dataset, predictions, print_scores=True
            )
            dev_scores.append(dev_score)

        if algorithm in ['perceptron']:
            logger.info(
                '\t'.join(
                    [
                        'Epoch: %d' % (epoch + 1),
                        'Mistakes: %d/%d (%f)'
                        % (
                            num_mistakes,
                            num_total,
                            float(num_mistakes) / float(num_total),
                        ),
                        'Dev scores: %s'
                        % ' '.join(
                            ["%.5g" % (100 * score) for score in dev_scores]
                        ),
                    ]
                )
            )
        else:
            sq_norm = self.classifier.model.weights.squared_norm()
            regularization_value = (
                0.5
                * lambda_coefficient
                * float(len(dataset))
                * weights.squared_norm()
            )
            logger.info(
                '\t'.join(
                    [
                        'Epoch: %d' % (epoch + 1),
                        'Cost: %f' % total_cost,
                        'Loss: %f' % total_loss,
                        'Reg: %f' % regularization_value,
                        'Loss+Reg: %f' % (total_loss + regularization_value),
                        'Norm: %f' % sq_norm,
                        'Dev scores: %s'
                        % ' '.join(
                            ["%.5g" % (100 * score) for score in dev_scores]
                        ),
                    ]
                )
            )

        # If this is the best model so far, save it as the default model.
        # Assume the metric to optimize is on the first dev set, the highest
        # the best.
        # TODO: replace by checkpointer functionality
        metric_value = dev_scores[0]
        if metric_value > self.best_metric_value:
            self.best_metric_value = metric_value
            self.checkpointer.check_in(
                self, self.best_metric_value, epoch=epoch
            )

        self.classifier.model.weights = weights
        self.classifier.model.averaged_weights = averaged_weights

    def save(self, output_directory):
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True)
        logging.info('Saving training state to {}'.format(output_directory))

        model_path = output_directory / const.MODEL_FILE

        self.classifier.model.save(
            str(model_path), feature_indices=self.classifier.feature_indices
        )
        self.classifier.save(str(model_path))

        return None
