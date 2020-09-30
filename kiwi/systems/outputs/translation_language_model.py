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
from collections import OrderedDict, defaultdict
from typing import Dict

from torch import nn
from torch.nn import CrossEntropyLoss, ModuleDict

from kiwi import constants as const
from kiwi.data.vocabulary import Vocabulary
from kiwi.metrics import CorrectMetric, ExpectedErrorMetric, PerplexityMetric
from kiwi.systems._meta_module import MetaModule
from kiwi.utils.io import BaseConfig
from kiwi.utils.tensors import replace_token


class MaskedWordOutput(nn.Module):
    def __init__(self, input_size, pad_idx, start_idx, stop_idx):
        super().__init__()

        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.stop_idx = stop_idx

        self.loss_fn = CrossEntropyLoss(reduction='sum', ignore_index=self.pad_idx)

    def forward(self, features_tensor):
        return features_tensor


@MetaModule.register_subclass
class TLMOutputs(MetaModule):
    class Config(BaseConfig):
        fine_tune: bool = False
        """Continue training an encoder on the post-edited text.
        Recommended if you have access to PE.
        Requires setting `system.data.train.input.pe`, `system.data.valid.input.pe`"""

        # pretrain: bool = False
        # """Train an encoder from scratch on parallel corpora.
        # Used to pretrain TLM models (like the Predictor).
        # """

    def __init__(
        self,
        inputs_dims: Dict[str, int],
        vocabs: Dict[str, Vocabulary],
        config: Config,
        pretraining: bool = False,
    ):
        super().__init__(config=config)

        self.inputs_dims = inputs_dims
        self.vocabs = OrderedDict()
        self.config = config
        self.pretraining = pretraining
        self._metrics = None

        self.masked_word_outputs = ModuleDict()

        if self.pretraining:
            if const.TARGET not in vocabs:
                raise ValueError(
                    f'Asked to pretrain the encoder (`pretrain`) but no '
                    f'vocabulary was provided for {const.TARGET}'
                )
            if const.TARGET_LOGITS not in self.inputs_dims:
                raise ValueError(
                    'Asked to pretrain the encoder (`pretrain`) but no '
                    'target data was provided'
                )
            self.masked_word_outputs[const.TARGET] = MaskedWordOutput(
                input_size=self.inputs_dims[const.TARGET_LOGITS],
                pad_idx=vocabs[const.TARGET].pad_id,
                start_idx=vocabs[const.TARGET].bos_id,
                stop_idx=vocabs[const.TARGET].eos_id,
            )
            self.vocabs[const.TARGET] = vocabs[const.TARGET]

        if self.config.fine_tune:
            # Target side; use PE for fine-tuning
            if const.PE not in vocabs:
                raise ValueError(
                    f'Asked to fine-tune the encoder (`fine_tune`) but no '
                    f'vocabulary was provided for {const.PE}'
                )
            if const.PE_LOGITS not in self.inputs_dims:
                raise ValueError(
                    'Asked to fine-tune the encoder (`fine_tune`) but no '
                    'post-edit (PE) data was provided'
                )
            self.masked_word_outputs[const.PE] = MaskedWordOutput(
                input_size=self.inputs_dims[const.PE_LOGITS],
                pad_idx=vocabs[const.PE].pad_id,
                start_idx=vocabs[const.PE].bos_id,
                stop_idx=vocabs[const.PE].eos_id,
            )
            self.vocabs[const.PE] = vocabs[const.PE]

    def forward(self, features, batch_inputs):
        outputs = OrderedDict()

        if const.PE_LOGITS in features and const.PE in self.masked_word_outputs:
            outputs[const.PE] = self.masked_word_outputs[const.PE](
                features[const.PE_LOGITS]
            )
        if const.TARGET_LOGITS in features and const.TARGET in self.masked_word_outputs:
            outputs[const.TARGET] = self.masked_word_outputs[const.TARGET](
                features[const.TARGET_LOGITS]
            )

        return outputs

    def loss(self, model_out, batch_outputs):
        loss_dict = OrderedDict()
        for output_side, layer in self.masked_word_outputs.items():
            if output_side in model_out:
                if output_side not in batch_outputs:
                    raise ValueError(
                        f'Model predicted {output_side} but true target is not in the '
                        f'batch.'
                    )
                target = batch_outputs[output_side].tensor
                # There are predictions for first/last element, so we want to
                #  mask them out if they are BOS and EOS tokens.
                target = replace_token(target, layer.start_idx, layer.pad_idx)
                target = replace_token(target, layer.stop_idx, layer.pad_idx)
                # Predicted Class must be in dim 1 for xentropyloss
                logits = model_out[output_side]
                logits = logits.transpose(1, 2)

                loss_dict[output_side] = layer.loss_fn(logits, target)

        loss_dict[const.LOSS] = sum(loss.sum() for _, loss in loss_dict.items())
        return loss_dict

    def metrics_step(self, batch, model_out, loss_dict):
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[metric.name] = metric.step(
                model_out=model_out, batch=batch, losses=loss_dict
            )
        return metrics_dict

    def metrics_end(self, steps, prefix=''):
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
    def metrics(self):
        if self._metrics is None:
            metrics = []
            for output_side, layer in self.masked_word_outputs.items():
                metrics.append(PerplexityMetric(output_side))
                metrics.append(
                    ExpectedErrorMetric(output_side, labels=self.labels(output_side))
                )
                metrics.append(
                    CorrectMetric(output_side, labels=self.labels(output_side))
                )
            self._metrics = metrics
        return self._metrics

    def labels(self, field):
        return [
            label
            for label in self.vocabs[field].itos
            if label not in self.vocabs[field].specials
        ]
