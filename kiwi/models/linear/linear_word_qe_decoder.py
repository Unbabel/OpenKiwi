"""Decoder for word-level quality estimation."""

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

from .sequence_parts import SequenceBigramPart, SequenceUnigramPart
from .structured_decoder import StructuredDecoder


def logzero():
    """Return log of zero."""
    return -np.inf


class LinearWordQEDecoder(StructuredDecoder):
    """A decoder for word-level quality estimation."""

    def __init__(
        self, estimator, cost_false_positives=0.5, cost_false_negatives=0.5
    ):
        StructuredDecoder.__init__(self)
        self.estimator = estimator
        self.cost_false_positives = cost_false_positives
        self.cost_false_negatives = cost_false_negatives

    def decode_mira(
        self, instance, parts, scores, gold_outputs, old_mira=False
    ):
        """Cost-augmented decoder. Allows a compromise between precision and
        recall. In general:
        p = a - (a+b)*z0
        q = b*sum(z0)
        p'*z + q = a*sum(z) - (a+b)*z0'*z + b*sum(z0)
                 = a*(1-z0)'*z + b*(1-z)'*z0
        a => penalty for predicting 1 when it is 0 (FP)
        b => penalty for predicting 0 when it is 1 (FN)

        F1: a = 0.5, b = 0.5
        recall: a = 0, b = 1"""

        a = self.cost_false_positives
        b = self.cost_false_negatives

        # Allow multiple bad labels.
        bad = []
        for label in self.estimator.labels:
            coarse_label = self.estimator.get_coarse_label(label)
            if coarse_label == 'BAD':
                bad.append(self.estimator.labels[label])
        bad = set(bad)
        index_parts = [
            i
            for i in range(len(parts))
            if isinstance(parts[i], SequenceUnigramPart)
            and parts[i].label in bad
        ]
        p = np.zeros(len(parts))
        p[index_parts] = a - (a + b) * gold_outputs[index_parts]
        q = b * np.ones(len(gold_outputs[index_parts])).dot(
            gold_outputs[index_parts]
        )
        if old_mira:
            predicted_outputs = self.decode(instance, parts, scores)
        else:
            scores_cost = scores + p
            predicted_outputs = self.decode(instance, parts, scores_cost)
        cost = p.dot(predicted_outputs) + q
        loss = cost + scores.dot(predicted_outputs - gold_outputs)
        return predicted_outputs, cost, loss

    def decode(self, instance, parts, scores):
        """Decoder. Return the most likely sequence of OK/BAD labels."""
        if self.estimator.use_bigrams:
            return self.decode_with_bigrams(instance, parts, scores)
        else:
            return self.decode_with_unigrams(instance, parts, scores)

    def decode_with_unigrams(self, instance, parts, scores):
        """Decoder for a non-sequential model (unigrams only)."""
        predicted_output = np.zeros(len(scores))
        parts_by_index = [[] for _ in range(instance.num_words())]
        for r, part in enumerate(parts):
            parts_by_index[part.index].append(r)

        for i in range(instance.num_words()):
            num_labels = len(parts_by_index[i])
            label_scores = np.zeros(num_labels)
            predicted_for_word = [0] * num_labels
            for k, r in enumerate(parts_by_index[i]):
                label_scores[k] = scores[r]
            best = np.argmax(label_scores)
            predicted_for_word[best] = 1.0
            r = parts_by_index[i][best]
            predicted_output[r] = 1.0

        return predicted_output

    def decode_with_bigrams(self, instance, parts, scores):
        """Decoder for a sequential model (with bigrams)."""
        num_labels = len(self.estimator.labels)
        num_words = instance.num_words()
        initial_scores = np.zeros(num_labels)
        transition_scores = np.zeros((num_words - 1, num_labels, num_labels))
        final_scores = np.zeros(num_labels)
        emission_scores = np.zeros((num_words, num_labels))

        indexed_unigram_parts = [{} for _ in range(num_words)]
        indexed_bigram_parts = [{} for _ in range(num_words + 1)]

        for r, part in enumerate(parts):
            if isinstance(part, SequenceUnigramPart):
                indexed_unigram_parts[part.index][part.label] = r
                emission_scores[part.index, part.label] = scores[r]
            elif isinstance(part, SequenceBigramPart):
                indexed_bigram_parts[part.index][
                    (part.label, part.previous_label)
                ] = r
                if part.previous_label < 0:
                    assert part.index == 0
                    initial_scores[part.label] = scores[r]
                elif part.label < 0:
                    assert part.index == num_words
                    final_scores[part.previous_label] = scores[r]
                else:
                    transition_scores[
                        part.index - 1, part.label, part.previous_label
                    ] = scores[r]
            else:
                raise NotImplementedError

        best_path, _ = self.run_viterbi(
            initial_scores, transition_scores, final_scores, emission_scores
        )

        predicted_output = np.zeros(len(scores))
        previous_label = -1
        for i, label in enumerate(best_path):
            r = indexed_unigram_parts[i][label]
            predicted_output[r] = 1.0
            r = indexed_bigram_parts[i][(label, previous_label)]
            predicted_output[r] = 1.0
            previous_label = label
        r = indexed_bigram_parts[num_words][(-1, previous_label)]
        predicted_output[r] = 1.0

        return predicted_output

    def run_viterbi(
        self, initial_scores, transition_scores, final_scores, emission_scores
    ):
        """Computes the viterbi trellis for a given sequence.
        Receives:
        - Initial scores: (num_states) array
        - Transition scores: (length-1, num_states, num_states) array
        - Final scores: (num_states) array
        - Emission scores: (length, num_states) array."""

        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Variables storing the Viterbi scores.
        viterbi_scores = np.zeros([length, num_states]) + logzero()

        # Variables storing the paths to backtrack.
        viterbi_paths = -np.ones([length, num_states], dtype=int)

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Initialization.
        viterbi_scores[0, :] = emission_scores[0, :] + initial_scores

        # Viterbi loop.
        for pos in range(1, length):
            for current_state in range(num_states):
                viterbi_scores[pos, current_state] = np.max(
                    viterbi_scores[pos - 1, :]
                    + transition_scores[pos - 1, current_state, :]
                )
                viterbi_scores[pos, current_state] += emission_scores[
                    pos, current_state
                ]
                viterbi_paths[pos, current_state] = np.argmax(
                    viterbi_scores[pos - 1, :]
                    + transition_scores[pos - 1, current_state, :]
                )
        # Termination.
        assert len(viterbi_scores[length - 1, :] + final_scores)
        best_score = np.max(viterbi_scores[length - 1, :] + final_scores)
        best_path[length - 1] = np.argmax(
            viterbi_scores[length - 1, :] + final_scores
        )

        # Backtrack.
        for pos in range(length - 2, -1, -1):
            best_path[pos] = viterbi_paths[pos + 1, best_path[pos + 1]]

        return best_path, best_score
