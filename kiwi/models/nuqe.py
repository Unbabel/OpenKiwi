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
from kiwi.models.model import Model
from kiwi.models.quetch import QUETCH
from kiwi.models.utils import make_loss_weights


@Model.register_subclass
class NuQE(QUETCH):
    """Neural Quality Estimation (NuQE) model for word level quality
    estimation."""

    title = 'NuQE'

    def __init__(self, vocabs, **kwargs):

        self.source_emb = None
        self.target_emb = None
        self.linear_1 = None
        self.linear_2 = None
        self.linear_3 = None
        self.linear_4 = None
        self.linear_5 = None
        self.linear_6 = None
        self.linear_out = None
        self.embeddings_dropout = None
        self.dropout = None
        self.gru1 = None
        self.gru2 = None
        self.is_built = False
        super().__init__(vocabs, **kwargs)

    def build(self, source_vectors=None, target_vectors=None):
        nb_classes = self.config.nb_classes
        # FIXME: Remove dependency on magic number
        weight = make_loss_weights(
            nb_classes, const.BAD_ID, self.config.bad_weight
        )

        self._loss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=self.config.tags_pad_id, reduction='sum'
        )

        # Embeddings layers:
        self._build_embeddings(source_vectors, target_vectors)

        feature_set_size = (
            self.config.source_embeddings_size
            + self.config.target_embeddings_size
        ) * self.config.window_size

        l1_dim = self.config.hidden_sizes[0]
        l2_dim = self.config.hidden_sizes[1]
        l3_dim = self.config.hidden_sizes[2]
        l4_dim = self.config.hidden_sizes[3]

        nb_classes = self.config.nb_classes
        dropout = self.config.dropout

        # Linear layers
        self.linear_1 = nn.Linear(feature_set_size, l1_dim)
        self.linear_2 = nn.Linear(l1_dim, l1_dim)
        self.linear_3 = nn.Linear(2 * l2_dim, l2_dim)
        self.linear_4 = nn.Linear(l2_dim, l2_dim)
        self.linear_5 = nn.Linear(2 * l2_dim, l3_dim)
        self.linear_6 = nn.Linear(l3_dim, l4_dim)

        # Output layer
        self.linear_out = nn.Linear(l4_dim, nb_classes)

        # Recurrent Layers
        self.gru_1 = nn.GRU(
            l1_dim, l2_dim, bidirectional=True, batch_first=True
        )
        self.gru_2 = nn.GRU(
            l2_dim, l2_dim, bidirectional=True, batch_first=True
        )

        # Dropout after linear layers
        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

        # Explicit initializations
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.xavier_uniform_(self.linear_5.weight)
        nn.init.xavier_uniform_(self.linear_6.weight)
        # nn.init.xavier_uniform_(self.linear_out)
        nn.init.constant_(self.linear_1.bias, 0.0)
        nn.init.constant_(self.linear_2.bias, 0.0)
        nn.init.constant_(self.linear_3.bias, 0.0)
        nn.init.constant_(self.linear_4.bias, 0.0)
        nn.init.constant_(self.linear_5.bias, 0.0)
        nn.init.constant_(self.linear_6.bias, 0.0)
        # nn.init.constant_(self.linear_out.bias, 0.)

        self.is_built = True

    @staticmethod
    def fieldset(*args, **kwargs):
        return build_fieldset(*args, **kwargs)

    @staticmethod
    def from_options(vocabs, opts):
        model = NuQE(
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
        h_source = self.embeddings_dropout(h_source)

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
        h_target = self.embeddings_dropout(h_target)

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
        h = self.dropout_in(h)

        #
        # First linears
        #
        # (bs, ts, 2 * window * emb) -> (bs, ts, l1_dim)
        h = F.relu(self.linear_1(h))

        # (bs, ts, l1_dim) -> (bs, ts, l1_dim)
        h = F.relu(self.linear_2(h))

        #
        # First recurrent
        #
        # (bs, ts, l1_dim) -> (bs, ts, l1_dim)
        h, _ = self.gru_1(h)

        #
        # Second linears
        #
        # (bs, ts, l1_dim) -> (bs, ts, l2_dim)
        h = F.relu(self.linear_3(h))

        # (bs, ts, l2_dim) -> (bs, ts, l2_dim)
        h = F.relu(self.linear_4(h))

        #
        # Second recurrent
        #
        # (bs, ts, l2_dim) -> (bs, ts, l2_dim)
        h, _ = self.gru_2(h)

        #
        # Third linears
        #
        # (bs, ts, l1_dim) -> (bs, ts, l3_dim)
        h = F.relu(self.linear_5(h))

        # (bs, ts, l3_dim) -> (bs, ts, l4_dim)
        h = F.relu(self.linear_6(h))
        h = self.dropout_out(h)

        #
        # Output layer
        #
        # (bs, ts, hs) -> (bs, ts, 2)
        h = self.linear_out(h)
        # h = F.log_softmax(h, dim=-1)

        outputs = OrderedDict()

        if self.config.predict_target:
            outputs[const.TARGET_TAGS] = h
        if self.config.predict_gaps:
            outputs[const.GAP_TAGS] = h
        if self.config.predict_source:
            outputs[const.SOURCE_TAGS] = h

        return outputs
