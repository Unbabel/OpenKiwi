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
import logging
from collections import OrderedDict, defaultdict
from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import ModuleDict, Sigmoid

from kiwi import constants as const
from kiwi.data.batch import MultiFieldBatch
from kiwi.data.vocabulary import Vocabulary
from kiwi.metrics import (
    CorrectMetric,
    F1MultMetric,
    MatthewsMetric,
    PearsonMetric,
    RMSEMetric,
    SpearmanMetric,
)
from kiwi.metrics.metrics import Metric
from kiwi.modules.sentence_level_output import (
    BinarySentenceScore,
    SentenceFromLogits,
    SentenceScoreDistribution,
    SentenceScoreRegression,
)
from kiwi.modules.word_level_output import GapTagsOutput, WordLevelOutput
from kiwi.systems._meta_module import MetaModule
from kiwi.utils.io import BaseConfig
from kiwi.utils.tensors import make_classes_loss_weights

logger = logging.getLogger(__name__)


class WordLevelConfig(BaseConfig):
    target: bool = False
    'Train or predict target tags'

    gaps: bool = False
    'Train or predict gap tags'

    source: bool = False
    'Train or predict source tags'

    class_weights: Dict[str, Dict[str, float]] = {
        const.TARGET_TAGS: {const.BAD: 3.0},
        const.GAP_TAGS: {const.BAD: 3.0},
        const.SOURCE_TAGS: {const.BAD: 3.0},
    }
    'Relative weight for labels on each output side.'


class SentenceLevelConfig(BaseConfig):
    hter: bool = False
    """Predict Sentence Level Scores.
    Requires the appropriate input files (usually with HTER)."""

    use_distribution: bool = False  # Old sentence_ll
    """Use probabilistic Loss for sentence scores instead of squared error.
    If set (requires `hter` to also be set), the model will output mean and variance
    of a truncated Gaussian distribution over the interval [0, 1], and use the NLL
    of ground truth scores as the loss.
    This seems to improve performance, and gives you uncertainty
    estimates for sentence level predictions as a byproduct.
    """

    binary: bool = False  # Old binary_level
    """Predict Binary Label for each sentence, indicating hter == 0.0.
    Requires the appropriate input files (usually with HTER)."""


@MetaModule.register_subclass
class QEOutputs(MetaModule):
    class Config(BaseConfig):
        word_level: WordLevelConfig = WordLevelConfig()
        sentence_level: SentenceLevelConfig = SentenceLevelConfig()
        sentence_loss_weight: float = 1.0
        'Multiplier for sentence_level loss weight.'

        dropout: float = 0.0
        use_final_sigmoid: bool = False
        n_layers_output: int = 3

    def __init__(self, inputs_dims, vocabs: Dict[str, Vocabulary], config: Config):
        super().__init__(config=config)

        self.inputs_dims = inputs_dims
        self.config = config
        self.vocabs = OrderedDict()
        self._metrics = None

        self.word_outputs = ModuleDict()

        tags_config = [
            (self.config.word_level.target, const.TARGET_TAGS),
            (self.config.word_level.source, const.SOURCE_TAGS),
            (self.config.word_level.gaps, const.GAP_TAGS),
        ]
        tags_sides = [tag for predict_tag, tag in tags_config if predict_tag]
        for tag_side in tags_sides:
            if tag_side not in vocabs:
                raise KeyError(
                    f'Asked to output {tag_side} but there is no vocabulary for it.'
                )
        if const.TARGET_TAGS in vocabs and self.config.word_level.target:
            class_weights = make_classes_loss_weights(
                vocab=vocabs[const.TARGET_TAGS],
                label_weights=self.config.word_level.class_weights[const.TARGET_TAGS],
            )
            self.word_outputs[const.TARGET_TAGS] = WordLevelOutput(
                input_size=self.inputs_dims[const.TARGET],
                output_size=vocabs[const.TARGET_TAGS].net_length(),
                pad_idx=vocabs[const.TARGET_TAGS].pad_id,
                class_weights=class_weights,
                remove_first=vocabs[const.TARGET].bos_id,
                remove_last=vocabs[const.TARGET].eos_id,
            )
            self.vocabs[const.TARGET_TAGS] = vocabs[const.TARGET_TAGS]
        if const.GAP_TAGS in vocabs and self.config.word_level.gaps:
            class_weights = make_classes_loss_weights(
                vocab=vocabs[const.GAP_TAGS],
                label_weights=self.config.word_level.class_weights[const.GAP_TAGS],
            )
            self.word_outputs[const.GAP_TAGS] = GapTagsOutput(
                input_size=self.inputs_dims[const.TARGET],
                output_size=vocabs[const.GAP_TAGS].net_length(),
                pad_idx=vocabs[const.GAP_TAGS].pad_id,
                class_weights=class_weights,
                remove_first=vocabs[const.TARGET].bos_id,
                remove_last=vocabs[const.TARGET].eos_id,
            )
            self.vocabs[const.GAP_TAGS] = vocabs[const.GAP_TAGS]
        if const.SOURCE_TAGS in vocabs and self.config.word_level.source:
            class_weights = make_classes_loss_weights(
                vocab=vocabs[const.SOURCE_TAGS],
                label_weights=self.config.word_level.class_weights[const.SOURCE_TAGS],
            )
            self.word_outputs[const.SOURCE_TAGS] = WordLevelOutput(
                input_size=self.inputs_dims[const.SOURCE],
                output_size=vocabs[const.SOURCE_TAGS].net_length(),
                pad_idx=vocabs[const.SOURCE_TAGS].pad_id,
                class_weights=class_weights,
                remove_first=vocabs[const.SOURCE].bos_id,
                remove_last=vocabs[const.SOURCE].eos_id,
            )
            self.vocabs[const.SOURCE_TAGS] = vocabs[const.SOURCE_TAGS]

        # Sentence level
        self.sentence_outputs = ModuleDict()

        if self.config.sentence_level.hter:
            if False:  # FIXME: add flag for regressing over average of word predictions
                self.sentence_outputs[const.SENTENCE_SCORES] = SentenceFromLogits()
            else:
                if const.TARGET_SENTENCE in self.inputs_dims:
                    input_size = self.inputs_dims[const.TARGET_SENTENCE]
                else:
                    input_size = self.inputs_dims[const.TARGET]
                if self.config.sentence_level.use_distribution:
                    sentence_scores = SentenceScoreDistribution(input_size=input_size)
                else:
                    sentence_scores = SentenceScoreRegression(
                        input_size=input_size,
                        num_layers=self.config.n_layers_output,
                        final_activation=Sigmoid
                        if self.config.use_final_sigmoid
                        else None,
                    )
                self.sentence_outputs[const.SENTENCE_SCORES] = sentence_scores
        # Binary sentence level
        if self.config.sentence_level.binary:
            if const.TARGET_SENTENCE in self.inputs_dims:
                input_size = self.inputs_dims[const.TARGET_SENTENCE]
            else:
                input_size = self.inputs_dims[const.TARGET]
            self.sentence_outputs[const.BINARY] = BinarySentenceScore(
                input_size=input_size
            )

    def forward(
        self, features: Dict[str, Tensor], batch_inputs: MultiFieldBatch
    ) -> Dict[str, Tensor]:
        outputs = OrderedDict()

        if self.config.word_level.target:
            if const.TARGET_TAGS in self.word_outputs and const.TARGET in features:
                outputs[const.TARGET_TAGS] = self.word_outputs[const.TARGET_TAGS](
                    features[const.TARGET], batch_inputs
                )
            elif const.TARGET_TAGS not in self.word_outputs:
                logger.warning(
                    f'Asked to output {const.TARGET_TAGS} but model has no layers for '
                    f'it; turning it off now.'
                )
                self.config.word_level.target = False
            else:
                logger.warning(
                    f'Asked to output {const.TARGET_TAGS} but no features for '
                    f'{const.TARGET} were provided'
                )
        if self.config.word_level.gaps:
            if const.GAP_TAGS in self.word_outputs and const.TARGET in features:
                outputs[const.GAP_TAGS] = self.word_outputs[const.GAP_TAGS](
                    features[const.TARGET], batch_inputs
                )
            elif const.GAP_TAGS not in self.word_outputs:
                logger.warning(
                    f'Asked to output {const.GAP_TAGS} but model has no layers for it; '
                    f'turning if off now.'
                )
                self.config.word_level.gaps = False
            else:
                logger.warning(
                    f'Asked to output {const.GAP_TAGS} but no features for '
                    f'{const.TARGET} were provided'
                )
        if self.config.word_level.source:
            if const.SOURCE_TAGS in self.word_outputs and const.SOURCE in features:
                outputs[const.SOURCE_TAGS] = self.word_outputs[const.SOURCE_TAGS](
                    features[const.SOURCE], batch_inputs
                )
            elif const.SOURCE_TAGS not in self.word_outputs:
                logger.warning(
                    f'Asked to output {const.SOURCE_TAGS} but model has no layers for '
                    f'it; turning it off now.'
                )
                self.config.word_level.source = False
            else:
                logger.warning(
                    f'Asked to output {const.SOURCE_TAGS} but no features for '
                    f'{const.SOURCE} were provided.'
                )

        # Sentence score and binary score prediction
        if self.config.sentence_level.hter:
            if False:  # FIXME: add flag for predicting from logits average
                _, lengths, *_ = batch_inputs[const.TARGET]
                sentence_score = self.sentence_pred(outputs[const.TARGET_TAGS], lengths)
                outputs[const.SENTENCE_SCORES] = sentence_score
            else:
                if const.SENTENCE_SCORES in self.sentence_outputs and (
                    const.TARGET_SENTENCE in features or const.TARGET in features
                ):
                    sentence_features = features.get(const.TARGET_SENTENCE)
                    if sentence_features is None:
                        sentence_features = features[const.TARGET][:, 0]
                    sentence_scores = self.sentence_outputs[const.SENTENCE_SCORES](
                        sentence_features, batch_inputs
                    )
                    outputs[const.SENTENCE_SCORES] = sentence_scores
                elif const.SENTENCE_SCORES not in self.sentence_outputs:
                    logger.warning(
                        f'Asked to output {const.SENTENCE_SCORES} but model has no '
                        f'layers for it; turning it off now.'
                    )
                    self.config.sentence_level.hter = False
                else:
                    logger.warning(
                        f'Asked to output {const.SENTENCE_SCORES} but no features for '
                        f'{const.TARGET_SENTENCE} or for {const.TARGET} were provided.'
                    )

        if self.config.sentence_level.binary:
            if const.BINARY in self.sentence_outputs and (
                const.TARGET_SENTENCE in features or const.TARGET in features
            ):
                sentence_features = features.get(const.TARGET_SENTENCE)
                if sentence_features is None:
                    sentence_features = features[const.TARGET][:, 0]
                outputs[const.BINARY] = self.sentence_outputs[const.BINARY](
                    sentence_features, batch_inputs
                )
            elif const.BINARY not in self.sentence_outputs:
                logger.warning(
                    f'Asked to output {const.BINARY} but model has no layers for it; '
                    f'turning it off now.'
                )
                self.config.sentence_level.binary = False
            else:
                logger.warning(
                    f'Asked to output {const.BINARY} but no features for '
                    f'{const.TARGET_SENTENCE} or for {const.TARGET} were provided.'
                )

        return outputs

    def loss(
        self, model_out: Dict[str, Tensor], batch: MultiFieldBatch
    ) -> Dict[str, Tensor]:
        loss_dict = self.word_losses(model_out, batch)

        loss_sent_dict = self.sentence_losses(model_out, batch)
        for name, loss_value in loss_sent_dict.items():
            loss_dict[name] = loss_value * self.config.sentence_loss_weight

        loss_dict[const.LOSS] = sum(loss.sum() for _, loss in loss_dict.items())
        return loss_dict

    def word_losses(self, model_out: Dict[str, Tensor], batch_outputs: MultiFieldBatch):
        """Compute sequence tagging loss."""
        word_loss = OrderedDict()
        for tag, layer in self.word_outputs.items():
            if tag in model_out:
                if tag not in batch_outputs:
                    raise ValueError(
                        f'Model predicted {tag} but true target is not in the batch.'
                    )
                logits = model_out[tag]
                logits = logits.transpose(1, 2)
                y = batch_outputs[tag].tensor
                try:
                    word_loss[tag] = layer.loss_fn(logits, y)
                except ValueError as e:
                    raise ValueError(f'with {tag}: {e}')
        return word_loss

    def sentence_losses(
        self, model_out: Dict[str, Tensor], batch_outputs: MultiFieldBatch
    ):
        """Compute sentence score loss."""
        sent_loss = OrderedDict()
        for label, layer in self.sentence_outputs.items():
            if label in model_out:
                if label not in batch_outputs:
                    raise ValueError(
                        f'Model predicted {label} but true target is not in the batch.'
                    )
                prediction = model_out[label]
                y = batch_outputs[label]
                sent_loss[label] = layer.loss_fn(prediction, y)
        return sent_loss

    def metrics_step(
        self,
        batch: MultiFieldBatch,
        model_out: Dict[str, Tensor],
        loss_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[metric.name] = metric.step(
                model_out=model_out, batch=batch, losses=loss_dict
            )
        return metrics_dict

    def metrics_end(self, steps: List[Dict[str, Tensor]], prefix=''):
        metrics_steps = defaultdict(list)
        for step in steps:
            for name, output in step.items():
                metrics_steps[name].append(output)
        metrics_steps = dict(metrics_steps)

        summary = {}
        for metric in self.metrics:
            summary.update(metric.compute(metrics_steps[metric.name], prefix=prefix))
        return summary

    @property
    def metrics(self) -> List[Metric]:
        if self._metrics is None:
            metrics = []
            if self.config.word_level.target and self.config.word_level.gaps:
                metrics += tag_metrics(
                    const.TARGET_TAGS,
                    const.GAP_TAGS,
                    prefix='WMT19_',
                    labels=self.labels(const.TARGET_TAGS),
                )
            if self.config.word_level.target:
                metrics += tag_metrics(
                    const.TARGET_TAGS, labels=self.labels(const.TARGET_TAGS)
                )
            if self.config.word_level.gaps:
                metrics += tag_metrics(
                    const.GAP_TAGS, labels=self.labels(const.GAP_TAGS)
                )
            if self.config.word_level.source:
                metrics += tag_metrics(
                    const.SOURCE_TAGS, labels=self.labels(const.SOURCE_TAGS)
                )

            if self.config.sentence_level.hter:
                metrics.append(PearsonMetric(const.SENTENCE_SCORES, prefix=''))
                metrics.append(SpearmanMetric(const.SENTENCE_SCORES, prefix=''))
                metrics.append(RMSEMetric(const.SENTENCE_SCORES, prefix=''))
            if self.config.sentence_level.binary:
                metrics.append(
                    CorrectMetric(
                        const.BINARY,
                        prefix='binary_',
                        labels=self.labels(const.TARGET_TAGS),
                    )
                )
            # metrics.append(LogMetric(log_targets=[(const.LOSS, const.LOSS)]))
            self._metrics = metrics
        return self._metrics

    def labels(self, field: str) -> List[str]:
        return [
            label
            for label in self.vocabs[field].itos
            if label not in self.vocabs[field].specials
        ]

    def decode_outputs(
        self,
        model_out: Dict[str, Tensor],
        batch_inputs: MultiFieldBatch,
        positive_class_label: str = const.BAD,
    ) -> Dict[str, List]:
        outputs = self.decode_word_outputs(
            model_out, batch_inputs, positive_class_label
        )
        outputs.update(self.decode_sentence_outputs(model_out))
        return outputs

    def decode_word_outputs(
        self,
        model_out: Dict[str, Tensor],
        batch_inputs: MultiFieldBatch,
        positive_class_label: str = const.BAD,
    ) -> Dict[str, List]:
        outputs = {}

        tags_config = [
            (const.TARGET_TAGS, 'target'),
            (const.SOURCE_TAGS, 'source'),
            (const.GAP_TAGS, 'target'),
        ]
        for key, input_side in tags_config:
            if key in model_out:
                # Models are assumed to return logits, not probabilities
                logits = model_out[key]
                probs = torch.softmax(logits, dim=-1)

                # Get string labels
                predicted_labels = probs.argmax(dim=-1, keepdim=False).tolist()

                # Get BAD probability
                class_index = torch.tensor(
                    [self.vocabs[key].token_to_id(positive_class_label)],
                    device=probs.device,
                    dtype=torch.long,
                )
                class_probs = probs.index_select(-1, class_index)
                class_probs = class_probs.squeeze(-1).tolist()

                # Convert into the right number of tokens per sample
                # Get lengths so we can unmask predictions and get rid of pads
                lengths = batch_inputs[input_side].number_of_tokens.clone()
                if key == const.GAP_TAGS:
                    lengths += 1  # Append one extra token

                for i, sample in enumerate(class_probs):
                    class_probs[i] = sample[: lengths[i]]
                for i, sample in enumerate(predicted_labels):
                    predicted_labels[i] = [
                        self.vocabs[key].id_to_token(x) for x in sample[: lengths[i]]
                    ]

                outputs[key] = class_probs
                outputs[f'{key}_labels'] = predicted_labels

        return outputs

    @staticmethod
    def decode_sentence_outputs(model_out: Dict[str, Tensor]) -> Dict[str, List]:
        outputs = {}

        if const.SENTENCE_SCORES in model_out:
            sentence_scores = model_out[const.SENTENCE_SCORES]
            if isinstance(sentence_scores, tuple):
                # By convention, first element are scores, the rest are extra data
                #  specific to that layer.
                #  E.g., here the rest are mean and std)
                extras = torch.stack(sentence_scores[1]).T.tolist()
                outputs[f'{const.SENTENCE_SCORES}_extras'] = extras
                sentence_scores = sentence_scores[0]
            outputs[const.SENTENCE_SCORES] = sentence_scores.tolist()
        if const.BINARY in model_out:
            logits = model_out[const.BINARY]
            probs = torch.softmax(logits, dim=-1)
            class_probs = probs[..., 0]  # FIXME: shouldn't this 0 be 1, for BAD?
            outputs[const.BINARY] = class_probs.tolist()

        return outputs


def tag_metrics(*targets, prefix=None, labels=None):
    metrics = [
        F1MultMetric(*targets, prefix=prefix, labels=labels),
        MatthewsMetric(*targets, prefix=prefix, labels=labels),
        # ThresholdCalibrationMetric(*targets, prefix=prefix),
        CorrectMetric(*targets, prefix=prefix, labels=labels),
    ]
    return metrics
