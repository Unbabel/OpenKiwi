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
