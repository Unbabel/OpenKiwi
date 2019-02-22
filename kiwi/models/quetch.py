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

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from kiwi import constants as const
from kiwi.data.fieldsets.quetch import build_fieldset
from kiwi.metrics import CorrectMetric, F1Metric, LogMetric
from kiwi.models.model import Model, ModelConfig
from kiwi.models.utils import align_tensor, convolve_tensor, make_loss_weights


class QUETCHConfig(ModelConfig):
    def __init__(
        self,
        vocabs,
        predict_target=True,
        predict_gaps=False,
        predict_source=False,
        source_embeddings_size=50,
        target_embeddings_size=50,
        hidden_sizes=None,
        bad_weight=3.0,
        window_size=10,
        max_aligned=5,
        dropout=0.4,
        embeddings_dropout=0.4,
        freeze_embeddings=False,
    ):
        super().__init__(vocabs)

        if hidden_sizes is None:
            hidden_sizes = [100]

        source_vectors = vocabs[const.SOURCE].vectors
        target_vectors = vocabs[const.TARGET].vectors
        if source_vectors is not None:
            source_embeddings_size = source_vectors.size(1)
        if target_vectors is not None:
            target_embeddings_size = target_vectors.size(1)

        self.source_embeddings_size = source_embeddings_size
        self.target_embeddings_size = target_embeddings_size

        self.bad_weight = bad_weight
        self.dropout = dropout
        self.embeddings_dropout = embeddings_dropout
        self.freeze_embeddings = freeze_embeddings
        # self.predict_side = predict_side

        # if predicting tags or source, default predict_target=true
        # doesn't make sense
        if predict_gaps or predict_source:
            predict_target = predict_target

        self.predict_target = predict_target
        self.predict_gaps = predict_gaps
        self.predict_source = predict_source

        self.window_size = window_size
        self.max_aligned = max_aligned
        self.hidden_sizes = hidden_sizes

        if const.SOURCE_TAGS in vocabs:
            self.tags_pad_id = vocabs[const.SOURCE_TAGS].stoi[const.PAD]
        elif const.GAP_TAGS in vocabs:
            self.tags_pad_id = vocabs[const.GAP_TAGS].stoi[const.PAD]
        else:
            self.tags_pad_id = vocabs[const.TARGET_TAGS].stoi[const.PAD]
        # FIXME: this might not correspond to reality (in vocabs)!
        self.nb_classes = len(const.LABELS)
        self.tag_bad_index = const.BAD_ID
        self.pad_token = const.PAD
        self.unaligned_idx = const.UNALIGNED_ID
        self.source_padding_idx = const.PAD_ID
        self.target_padding_idx = const.PAD_ID


@Model.register_subclass
class QUETCH(Model):
    """QUality Estimation from scraTCH (QUETCH) model.

    TODO: add references.

    """

    title = "QUETCH"

    def __init__(self, vocabs, **kwargs):

        super().__init__(vocabs=vocabs, ConfigCls=QUETCHConfig, **kwargs)

        self.source_emb = None
        self.target_emb = None
        self.embeddings_dropout = None
        self.linear = None
        self.dropout = None
        self.linear_out = None

        source_vectors = vocabs[const.SOURCE].vectors
        target_vectors = vocabs[const.TARGET].vectors
        self.build(source_vectors, target_vectors)

    @staticmethod
    def fieldset(*args, **kwargs):
        return build_fieldset(*args, **kwargs)

    @staticmethod
    def from_options(vocabs, opts):
        model = QUETCH(
            vocabs=vocabs,
            predict_target=opts.predict_target,
            predict_gaps=opts.predict_gaps,
            predict_source=opts.predict_source,
            source_embeddings_size=opts.source_embeddings_size,
            target_embeddings_size=opts.target_embeddings_size,
            hidden_sizes=opts.hidden_sizes,
            bad_weight=opts.bad_weight,
            window_size=opts.window_size,
            max_aligned=opts.max_aligned,
            dropout=opts.dropout,
            embeddings_dropout=opts.embeddings_dropout,
            freeze_embeddings=opts.freeze_embeddings,
        )
        return model

    def loss(self, model_out, target):
        if self.config.predict_source:
            output_name = const.SOURCE_TAGS
        elif self.config.predict_gaps:
            output_name = const.GAP_TAGS
        else:
            output_name = const.TARGET_TAGS

        # (bs*ts, nb_classes)
        probs = model_out[output_name]
        # (bs*ts, )
        y = getattr(target, output_name)

        predicted = probs.view(-1, self.config.nb_classes)
        y = y.view(-1)

        loss = self._loss(predicted, y)
        return {const.LOSS: loss}

    def _build_embeddings(self, source_vectors=None, target_vectors=None):
        # Embeddings layers:
        if source_vectors is not None:
            # source_embeddings_size = self.source_embeddings.size(1)
            self.source_emb = nn.Embedding(
                num_embeddings=source_vectors.size(0),
                embedding_dim=source_vectors.size(1),
                padding_idx=self.config.source_padding_idx,
                _weight=source_vectors,
            )
        else:
            self.source_emb = nn.Embedding(
                num_embeddings=self.config.source_vocab_size,
                embedding_dim=self.config.source_embeddings_size,
                padding_idx=self.config.source_padding_idx,
            )
        if target_vectors is not None:
            self.target_emb = nn.Embedding(
                num_embeddings=target_vectors.size(0),
                embedding_dim=target_vectors.size(1),
                padding_idx=self.config.target_padding_idx,
                _weight=target_vectors,
            )
        else:
            self.target_emb = nn.Embedding(
                num_embeddings=self.config.target_vocab_size,
                embedding_dim=self.config.target_embeddings_size,
                padding_idx=self.config.target_padding_idx,
            )
        if self.config.freeze_embeddings:
            self.source_emb.weight.requires_grad = False
            self.source_emb.bias.requires_grad = False
            self.target_emb.weight.requires_grad = False
            self.target_emb.bias.requires_grad = False

        self.embeddings_dropout = nn.Dropout(self.config.embeddings_dropout)

    def build(self, source_vectors=None, target_vectors=None):

        hidden_size = self.config.hidden_sizes[0]
        nb_classes = self.config.nb_classes
        dropout = self.config.dropout

        weight = make_loss_weights(
            nb_classes, const.BAD_ID, self.config.bad_weight
        )

        self._loss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=const.PAD_TAGS_ID
        )

        # Embeddings layers:
        self._build_embeddings(source_vectors, target_vectors)

        feature_set_size = (
            self.config.source_embeddings_size
            + self.config.target_embeddings_size
        ) * self.config.window_size

        self.linear = nn.Linear(feature_set_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, nb_classes)

        self.dropout = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear.bias, 0.0)
        torch.nn.init.constant_(self.linear_out.bias, 0.0)

        self.is_built = True

    def make_input(self, batch, side):
        target_input, target_lengths = getattr(batch, const.TARGET)
        source_input, source_lengths = getattr(batch, const.SOURCE)
        alignments = batch.alignments

        if self.config.predict_gaps and not self.config.predict_target:
            target_input = F.pad(
                target_input,
                pad=(0, 1),
                value=self.vocabs[const.TARGET].stoi[const.UNALIGNED],
            )
            source_input = F.pad(
                source_input,
                pad=(0, 1),
                value=self.vocabs[const.SOURCE].stoi[const.UNALIGNED],
            )

        target_input = convolve_tensor(
            target_input,
            self.config.window_size,
            self.config.target_padding_idx,
        )

        source_input = convolve_tensor(
            source_input,
            self.config.window_size,
            self.config.source_padding_idx,
        )

        if side == const.SOURCE_TAGS:
            alignments = [
                [alignment[::-1] for alignment in example_alignment]
                for example_alignment in alignments
            ]
            target_input, nb_alignments = align_tensor(
                target_input,
                alignments,
                self.config.max_aligned,
                self.config.unaligned_idx,
                self.config.target_padding_idx,
                pad_size=source_input.shape[1],
            )
        else:
            source_input, nb_alignments = align_tensor(
                source_input,
                alignments,
                self.config.max_aligned,
                self.config.unaligned_idx,
                self.config.source_padding_idx,
                pad_size=target_input.shape[1],
            )

        return target_input, source_input, nb_alignments

    def forward(self, batch):
        assert self.is_built

        if self.config.predict_source:
            align_side = const.SOURCE_TAGS
        else:
            align_side = const.TARGET_TAGS

        target_input, source_input, nb_alignments = self.make_input(
            batch, align_side
        )

        #
        # Source Branch
        #
        # (bs, ts, aligned, window) -> (bs, ts, aligned, window, emb)
        h_source = self.source_emb(source_input)

        if len(h_source.shape) == 5:
            # (bs, ts, aligned, window, emb) -> (bs, ts, window, emb)
            h_source = h_source.sum(2, keepdim=False) / nb_alignments.unsqueeze(
                -1
            ).unsqueeze(-1)

        # (bs, ts, window, emb) -> (bs, ts, window * emb)
        h_source = h_source.view(source_input.size(0), source_input.size(1), -1)

        #
        # Target Branch
        #
        # (bs, ts * window) -> (bs, ts * window, emb)
        h_target = self.target_emb(target_input)

        if len(h_target.shape) == 5:
            # (bs, ts, aligned, window, emb) -> (bs, ts, window, emb)
            h_target = h_target.sum(2, keepdim=False) / nb_alignments.unsqueeze(
                -1
            ).unsqueeze(-1)

        # (bs, ts * window, emb) -> (bs, ts, window * emb)
        h_target = h_target.view(target_input.size(0), target_input.size(1), -1)

        #
        # POS tags branches
        #
        feature_set = (h_source, h_target)

        #
        # Merge Branches
        #
        # (bs, ts, window * emb) -> (bs, ts, 2 * window * emb)
        h = torch.cat(feature_set, dim=-1)
        h = self.embeddings_dropout(h)
        # (bs, ts, 2 * window * emb) -> (bs, ts, hs)
        h = torch.tanh(self.linear(h))
        h = self.dropout(h)

        # (bs, ts, hs) -> (bs, ts, 2)
        h = self.linear_out(h)

        outputs = OrderedDict()

        if self.config.predict_target:
            outputs[const.TARGET_TAGS] = h
        if self.config.predict_gaps:
            outputs[const.GAP_TAGS] = h
        if self.config.predict_source:
            outputs[const.SOURCE_TAGS] = h

        return outputs

    @staticmethod
    def _unmask(tensor, mask):
        lengths = mask.int().sum(dim=-1)
        return [x[: lengths[i]] for i, x in enumerate(tensor)]

    def metrics(self):
        metrics = []

        if self.config.predict_target:
            metrics.append(
                F1Metric(
                    prefix=const.TARGET_TAGS,
                    target_name=const.TARGET_TAGS,
                    PAD=const.PAD_TAGS_ID,
                    labels=const.LABELS,
                )
            )
            metrics.append(
                CorrectMetric(
                    prefix=const.TARGET_TAGS,
                    target_name=const.TARGET_TAGS,
                    PAD=const.PAD_TAGS_ID,
                )
            )
        if self.config.predict_source:
            metrics.append(
                F1Metric(
                    prefix=const.SOURCE_TAGS,
                    target_name=const.SOURCE_TAGS,
                    PAD=const.PAD_TAGS_ID,
                    labels=const.LABELS,
                )
            )
            metrics.append(
                CorrectMetric(
                    prefix=const.SOURCE_TAGS,
                    target_name=const.SOURCE_TAGS,
                    PAD=const.PAD_TAGS_ID,
                )
            )
        if self.config.predict_gaps:
            metrics.append(
                F1Metric(
                    prefix=const.GAP_TAGS,
                    target_name=const.GAP_TAGS,
                    PAD=const.PAD_TAGS_ID,
                    labels=const.LABELS,
                )
            )
            metrics.append(
                CorrectMetric(
                    prefix=const.GAP_TAGS,
                    target_name=const.GAP_TAGS,
                    PAD=const.PAD_TAGS_ID,
                )
            )

        metrics.append(LogMetric(targets=[(const.LOSS, const.LOSS)]))

        return metrics

    def metrics_ordering(self):
        return max
