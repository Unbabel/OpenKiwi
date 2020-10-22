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
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn import ModuleDict

from kiwi import constants as const
from kiwi.data.vocabulary import Vocabulary
from kiwi.modules.token_embeddings import TokenEmbeddings
from kiwi.systems._meta_module import MetaModule
from kiwi.utils.io import BaseConfig
from kiwi.utils.tensors import convolve_tensor

logger = logging.getLogger(__name__)


class InputEmbeddingsConfig(BaseConfig):
    """Embeddings size for each input field, if they are not loaded."""

    source: TokenEmbeddings.Config = TokenEmbeddings.Config()
    target: TokenEmbeddings.Config = TokenEmbeddings.Config()
    source_pos: Optional[TokenEmbeddings.Config]
    target_pos: Optional[TokenEmbeddings.Config]


@MetaModule.register_subclass
class QUETCHEncoder(MetaModule):
    class Config(BaseConfig):
        window_size: int = 3
        """Size of sliding window."""

        embeddings: InputEmbeddingsConfig

    def __init__(
        self, vocabs: Dict[str, Vocabulary], config: Config, pre_load_model: bool = True
    ):
        super().__init__(config=config)

        self.embeddings = ModuleDict()
        self.embeddings[const.TARGET] = TokenEmbeddings(
            num_embeddings=len(vocabs[const.TARGET]),
            pad_idx=vocabs[const.TARGET].pad_id,
            config=config.embeddings.target,
            vectors=vocabs[const.TARGET].vectors,
        )
        self.embeddings[const.SOURCE] = TokenEmbeddings(
            num_embeddings=len(vocabs[const.SOURCE]),
            pad_idx=vocabs[const.SOURCE].pad_id,
            config=config.embeddings.source,
            vectors=vocabs[const.SOURCE].vectors,
        )

        total_size = sum(emb.size() for emb in self.embeddings.values())
        self._sizes = {
            const.TARGET: total_size * self.config.window_size,
            const.SOURCE: total_size * self.config.window_size,
        }

    @classmethod
    def input_data_encoders(cls, config: Config):
        return None  # Use defaults, i.e., TextEncoder

    def size(self, field=None):
        if field:
            return self._sizes[field]
        return self._sizes

    def forward(self, batch_inputs):
        target_emb = self.embeddings[const.TARGET](batch_inputs[const.TARGET])
        source_emb = self.embeddings[const.SOURCE](batch_inputs[const.SOURCE])

        if const.TARGET_POS in self.embeddings:
            pos_emb, _ = self.embeddings[const.TARGET_POS](
                batch_inputs[const.TARGET_POS]
            )
            target_emb = torch.cat((target_emb, pos_emb), dim=-1)
        if const.SOURCE_POS in self.embeddings:
            pos_emb, _ = self.embeddings[const.SOURCE_POS](
                batch_inputs[const.SOURCE_POS]
            )
            source_emb = torch.cat((source_emb, pos_emb), dim=-1)

        # (bs, source_steps, target_steps)
        matrix_alignments = batch_inputs[const.ALIGNMENTS]
        # Timesteps might actually be longer when the last words are not aligned
        pad = [0, 0, 0, 0]
        if matrix_alignments.size(1) < source_emb.size(1):
            pad[3] = source_emb.size(1) - matrix_alignments.size(1)
        if matrix_alignments.size(2) < target_emb.size(1):
            pad[1] = target_emb.size(1) - matrix_alignments.size(2)
        if any(pad):
            matrix_alignments = F.pad(matrix_alignments, pad=pad, value=0)

        h_target = convolve_tensor(
            target_emb,
            self.config.window_size,
            pad_value=self.embeddings[const.TARGET].pad_idx,
        )
        h_source = convolve_tensor(
            source_emb,
            self.config.window_size,
            pad_value=self.embeddings[const.SOURCE].pad_idx,
        )
        h_target = h_target.contiguous().view(h_target.shape[0], h_target.shape[1], -1)
        h_source = h_source.contiguous().view(h_source.shape[0], h_source.shape[1], -1)

        # Target side
        matrix_alignments_t = matrix_alignments.transpose(1, 2).float()
        # (bs, target_steps, source_steps) x (bs, source_steps, *)
        # -> (bs, target_steps, *)
        # Take the mean of aligned tokens
        h_source_to_target = torch.matmul(matrix_alignments_t, h_source)
        z = matrix_alignments_t.sum(dim=2, keepdim=True)
        z[z == 0] = 1.0
        h_source_to_target = h_source_to_target / z
        # h_source_to_target[h_source_to_target.sum(-1) == 0] = self.unaligned_source
        # assert torch.all(torch.eq(h_source_to_target, h_source_to_target))
        features_target = torch.cat((h_source_to_target, h_target), dim=-1)

        # Source side
        matrix_alignments = matrix_alignments.float()
        # (bs, source_steps, target_steps) x (bs, target_steps, *)
        # -> (bs, source_steps, *)
        # Take the mean of aligned tokens
        h_target_to_source = torch.matmul(matrix_alignments, h_target)
        z = matrix_alignments.sum(dim=2, keepdim=True)
        z[z == 0] = 1.0
        h_target_to_source = h_target_to_source / z
        # h_target_to_source[h_target_to_source.sum(-1) == 0] = self.unaligned_target
        features_source = torch.cat((h_source, h_target_to_source), dim=-1)

        # (bs, ts, window * emb) -> (bs, ts, 2 * window * emb)
        features = {const.TARGET: features_target, const.SOURCE: features_source}

        return features
