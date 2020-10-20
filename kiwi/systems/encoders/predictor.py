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
from collections import OrderedDict
from typing import Dict, Optional

import torch
from pydantic import validator
from torch import nn

from kiwi import constants as const
from kiwi.data.vocabulary import Vocabulary
from kiwi.modules.common.attention import Attention
from kiwi.modules.common.scorer import MLPScorer
from kiwi.modules.token_embeddings import TokenEmbeddings
from kiwi.systems._meta_module import MetaModule
from kiwi.systems.encoders.quetch import InputEmbeddingsConfig
from kiwi.utils.io import BaseConfig
from kiwi.utils.tensors import apply_packed_sequence, pad_zeros_around_timesteps

logger = logging.getLogger(__name__)


class DualSequencesEncoder(nn.Module):
    def __init__(
        self,
        input_size_a,
        input_size_b,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        _use_v0_buggy_strategy=False,
    ):
        super().__init__()
        self._use_v0_buggy_strategy = _use_v0_buggy_strategy  # Check doc in Config

        scorer = MLPScorer(hidden_size * 2, hidden_size * 2)
        self.attention = Attention(scorer)

        self.forward_backward_a = nn.LSTM(
            input_size=input_size_a,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.forward_b = nn.LSTM(
            input_size=input_size_b,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.backward_b = nn.LSTM(
            input_size=input_size_b,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.W2 = nn.Parameter(
            torch.zeros(output_size, output_size), requires_grad=True
        )

        self.V = nn.Parameter(
            torch.zeros(2 * input_size_b, 2 * output_size), requires_grad=True
        )
        self.C = nn.Parameter(
            torch.zeros(2 * hidden_size, 2 * output_size), requires_grad=True
        )
        self.S = nn.Parameter(
            torch.zeros(2 * hidden_size, 2 * output_size), requires_grad=True
        )

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, embeddings_a, lengths_a, mask_a, embeddings_b, lengths_b):
        if self._use_v0_buggy_strategy:
            embeddings_a = embeddings_a[:, :-2]
            mask_a = mask_a[:, 1:-1]
            lengths_a -= 2  # Equivalent to mask_a.sum(dim=1)
            lengths_b -= 2

        # Encode sequence A
        contexts_a, hidden = apply_packed_sequence(
            self.forward_backward_a, embeddings_a, lengths_a
        )

        # Encode sequence B
        forward_contexts, backward_contexts = self.contextualize_b(
            embeddings_b, lengths_b, hidden
        )
        post_QEFV = torch.cat([forward_contexts, backward_contexts], dim=-1)

        f = self.encode_b(
            embeddings_b, forward_contexts, backward_contexts, contexts_a, mask_a
        )

        return f, post_QEFV

    def contextualize_b(self, embeddings, lengths, hidden):
        h_forward, h_backward = self._split_hidden(hidden)
        # Note: hidden, h_forward and h_backward are not batch_first
        forward_contexts, _ = self.forward_b(embeddings, h_forward)
        reversed_embeddings = self._reverse_padded_seq(lengths, embeddings)
        backward_contexts, _ = self.backward_b(reversed_embeddings, h_backward)
        backward_contexts = self._reverse_padded_seq(lengths, backward_contexts)

        return forward_contexts, backward_contexts

    def encode_b(
        self,
        embeddings,
        forward_contexts,
        backward_contexts,
        contexts_a,
        attention_mask,
    ):
        """Encode sequence B.

        Build a feature vector for each position i using left context i-1 and right
        context i+1. In the original implementation, this resulted in a returned tensor
        with -2 timesteps (dim=1). We have now changed it to return the same number
        of timesteps as the input. The consequence is that callers now have to deal
        with BOS and EOS in a different way, but hopefully this new behaviour is more
        consistent and less surprising. The old behaviour can be forced by setting
        ``self._use_v0_buggy_strategy`` to True.
        """
        if not self._use_v0_buggy_strategy:
            # Pad inputs on both sides for retaining the original number of timesteps
            forward_contexts = pad_zeros_around_timesteps(forward_contexts)
            backward_contexts = pad_zeros_around_timesteps(backward_contexts)
            embeddings = pad_zeros_around_timesteps(embeddings)

        # For each position, concatenate left context i-1 and right context i+1
        # (bs, target_len, d) -> (bs, target_len-2, d*2)
        contexts_b = torch.cat(
            [forward_contexts[:, :-2], backward_contexts[:, 2:]], dim=-1
        )

        # For each position i, concatenate Embeddings i-1 and i+1
        context_b_embeddings = torch.cat(
            [embeddings[:, :-2], embeddings[:, 2:]], dim=-1
        )

        # Get Attention for all positions and stack (vectorized)
        attns, p_attns = self.attention(
            query=contexts_b,
            keys=contexts_a,
            values=contexts_a,
            mask=attention_mask.unsqueeze(1),
        )

        # Combine attention, embeddings and target context vectors
        S = torch.einsum('bsk,kl->bsl', [contexts_b, self.S])
        V = torch.einsum('bsj,jl->bsl', [context_b_embeddings, self.V])
        C = torch.einsum('bsi,il->bsl', [attns, self.C])
        t_tilde = S + V + C

        # Maxout with pooling size 2
        t, _ = torch.max(
            t_tilde.view(t_tilde.shape[0], t_tilde.shape[1], t_tilde.shape[-1] // 2, 2),
            dim=-1,
        )
        f = torch.einsum('oh,bso->bsh', [self.W2, t])

        return f

    @staticmethod
    def _reverse_padded_seq(lengths, sequence):
        """Reverse a batch of padded sequences of different length."""
        batch_size, max_length = sequence.shape[:-1]
        reversed_idx = []
        for i in range(batch_size * max_length):
            batch_id = i // max_length
            sent_id = i % max_length
            if sent_id < lengths[batch_id]:
                sent_id_rev = lengths[batch_id] - sent_id - 1
            else:
                sent_id_rev = sent_id  # Padding symbol, don't change order
            reversed_idx.append(max_length * batch_id + sent_id_rev)
        flat_sequence = sequence.contiguous().view(batch_size * max_length, -1)
        reversed_seq = flat_sequence[reversed_idx, :].view(*sequence.shape)
        return reversed_seq

    @staticmethod
    def _split_hidden(hidden):
        """Split hidden state into forward/backward parts."""
        h, c = hidden
        size = h.size(0)
        idx_forward = torch.arange(0, size, 2, dtype=torch.long)
        idx_backward = torch.arange(1, size, 2, dtype=torch.long)
        hidden_forward = (h[idx_forward], c[idx_forward])
        hidden_backward = (h[idx_backward], c[idx_backward])
        return hidden_forward, hidden_backward


@MetaModule.register_subclass
class PredictorEncoder(MetaModule):
    """Bidirectional Conditional Language Model

    Implemented after Kim et al 2017, see: http://www.statmt.org/wmt17/pdf/WMT63.pdf
    """

    class Config(BaseConfig):
        encode_source: bool = False

        hidden_size: int = 400
        """Size of hidden layers in LSTM."""

        rnn_layers: int = 3
        """Number of RNN layers in the Predictor."""

        dropout: float = 0.0

        share_embeddings: bool = False
        """Tie input and output embeddings for target."""

        out_embeddings_dim: Optional[int] = None
        """Word Embedding in Output layer."""

        use_mismatch_features: bool = False
        """Whether to use Alibaba's mismatch features."""

        embeddings: InputEmbeddingsConfig = InputEmbeddingsConfig()

        use_v0_buggy_strategy: bool = False
        """The Predictor implementation in Kiwi<=0.3.4 had a bug in applying the LSTM
        to encode source (it used lengths too short by 2) and in reversing the target
        embeddings for applying the backward LSTM (also short by 2). This flag is set
        to true when loading a saved model from those versions."""
        v0_start_stop: bool = False
        """Whether pre_qe_f_v is padded on both ends or
        post_qe_f_v is strip on both ends."""

        @validator('dropout', pre=True)
        def dropout_on_rnns(cls, v, values):
            if v > 0.0 and values['rnn_layers'] == 1:
                logger.info(
                    'Dropout on an RNN of one layer has no effect; setting it to zero.'
                )
                return 0.0
            return v

        @validator('use_mismatch_features', pre=True)
        def no_implementation(cls, v):
            if v:
                raise NotImplementedError('Not yet implemented')
            return False

    def __init__(
        self,
        vocabs: Dict[str, Vocabulary],
        config: Config,
        pretraining: bool = False,
        pre_load_model: bool = True,
    ):
        """
        Arguments:
            vocabs: dictionary Mapping Field Names to Vocabularies.
            config: a state dict of a PredictorConfig object.
            pretraining: set it to True when pretraining with parallel data.
            pre_load_model: not used
        """
        super().__init__(config=config)
        self.pretraining = pretraining

        # Input embeddings
        self.embeddings = nn.ModuleDict()
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
        self.vocabs = {
            const.TARGET: vocabs[const.TARGET],
            const.SOURCE: vocabs[const.SOURCE],
        }

        # Output embeddings
        self.output_embeddings = nn.ModuleDict()
        if self.config.share_embeddings:
            self.output_embeddings[const.TARGET] = self.embeddings[
                const.TARGET
            ].embedding
        else:
            self.output_embeddings[const.TARGET] = nn.Embedding(
                num_embeddings=self.embeddings[const.TARGET].num_embeddings,
                embedding_dim=self.config.out_embeddings_dim,
                padding_idx=self.embeddings[const.TARGET].pad_idx,
            )

        # Encoders
        self.encode_target = DualSequencesEncoder(
            input_size_a=self.embeddings[const.SOURCE].size(),
            input_size_b=self.embeddings[const.TARGET].size(),
            hidden_size=self.config.hidden_size,
            output_size=self.output_embeddings[const.TARGET].embedding_dim,
            num_layers=self.config.rnn_layers,
            dropout=self.config.dropout,
            _use_v0_buggy_strategy=self.config.use_v0_buggy_strategy,
        )

        output_dim = self.output_embeddings[const.TARGET].embedding_dim
        self.start_PreQEFV = nn.Parameter(
            torch.zeros(1, 1, output_dim), requires_grad=True
        )
        self.end_PreQEFV = nn.Parameter(
            torch.zeros(1, 1, output_dim), requires_grad=True
        )

        # total_size = sum(emb.size() for emb in self.embeddings.values())
        self._sizes = {
            const.TARGET: output_dim + 2 * self.config.hidden_size,
            # const.SOURCE: output_dim + 2 * self.config.hidden_size,
            const.TARGET_LOGITS: self.output_embeddings[const.TARGET].num_embeddings,
            const.PE_LOGITS: self.output_embeddings[const.TARGET].num_embeddings,
        }

        # For multi-tasking on source side
        # if self.config.encode_source:
        #     if self.config.share_embeddings:
        #         self.output_embeddings[const.SOURCE] = (
        #             self.embeddings[const.SOURCE].embedding
        #         )
        #     else:
        #         self.output_embeddings[const.SOURCE] = nn.Embedding(
        #             num_embeddings=self.embeddings[const.SOURCE].vocab_size(),
        #             embedding_dim=self.config.out_embeddings_dim,
        #             padding_idx=self.embeddings[const.SOURCE].pad_idx,
        #         )
        #
        #     self.encode_source = PredictorSourceEncoder(
        #         hidden_size=self.config.hidden_size
        #     )
        #
        #     # Don't initialize this (TODO: but check what happens)
        #     self.start_PreQEFV_source = nn.Parameter(
        #         torch.zeros(1, 1, self.output_embeddings[const.SOURCE].size(0)),
        #         requires_grad=True
        #     )
        #     self.end_PreQEFV_source = nn.Parameter(
        #         torch.zeros(1, 1, self.output_embeddings[const.SOURCE].size(0)),
        #         requires_grad=True
        #     )
        #     self._sizes[const.SOURCE] = (
        #         self.output_embeddings[const.SOURCE].size(0)\
        #         + 2 * self.config.hidden_size
        #     )

    @classmethod
    def input_data_encoders(cls, config: Config):
        return None  # Use defaults, i.e., TextEncoder

    def size(self, field=None):
        if field:
            return self._sizes[field]
        return self._sizes

    def forward(self, batch_inputs, include_target_logits=False):
        target_embeddings = self.embeddings[const.TARGET](batch_inputs[const.TARGET])
        source_embeddings = self.embeddings[const.SOURCE](batch_inputs[const.SOURCE])

        target_lengths = batch_inputs[const.TARGET].lengths
        source_lengths = batch_inputs[const.SOURCE].lengths

        source_attention_mask = batch_inputs[const.SOURCE].strict_masks

        f, post_qe_feature_vector = self.encode_target(
            source_embeddings,
            source_lengths,
            source_attention_mask,
            target_embeddings,
            target_lengths,
        )

        output_embeddings = self.output_embeddings[const.TARGET](
            batch_inputs[const.TARGET].tensor
        )
        if self.config.use_v0_buggy_strategy:
            pre_qe_feature_vector = torch.einsum(
                'bsh,bsh->bsh', [output_embeddings[:, 1:-1], f]
            )
            if self.config.v0_start_stop:
                start = self.start_PreQEFV.expand(output_embeddings.size(0), -1, -1)
                end = self.end_PreQEFV.expand(output_embeddings.size(0), -1, -1)
                pre_qe_feature_vector = torch.cat(
                    (start, pre_qe_feature_vector, end), dim=1
                )
            else:
                post_qe_feature_vector = post_qe_feature_vector[:, 1:-1]
        else:
            pre_qe_feature_vector = torch.einsum('bsh,bsh->bsh', [output_embeddings, f])
            # Using these learnable start and stop parameters seemed to help (according
            #  to Sony).
            start = self.start_PreQEFV.expand(output_embeddings.size(0), -1, -1)
            end = self.end_PreQEFV.expand(output_embeddings.size(0), -1, -1)
            pre_qe_feature_vector = torch.cat(
                (start, pre_qe_feature_vector[:, 1:-1], end), dim=1
            )

        features = torch.cat([pre_qe_feature_vector, post_qe_feature_vector], dim=-1)

        output_features = OrderedDict()
        output_features[const.TARGET] = features

        if include_target_logits or self.pretraining:
            logits = torch.einsum(
                'vh,bsh->bsv', [self.output_embeddings[const.TARGET].weight, f]
            )
            output_features[const.TARGET_LOGITS] = logits

        if const.PE in batch_inputs:
            pe_embeddings = self.embeddings[const.TARGET](batch_inputs[const.PE])
            pe_lengths = batch_inputs[const.PE].lengths
            f, _ = self.encode_target(
                source_embeddings,
                source_lengths,
                source_attention_mask,
                pe_embeddings,
                pe_lengths,
            )
            output_features[const.PE_LOGITS] = torch.einsum(
                'vh,bsh->bsv', [self.output_embeddings[const.TARGET].weight, f]
            )

        # if self.config.predict_source:
        #     f, backward_contexts, forward_contexts = self.encode_source(
        #         target_embeddings,
        #         target_lengths,
        #         target_attention_mask,
        #         source_embeddings,
        #         source_lengths,
        #     )
        #
        #     if output_logits:
        #         logits = torch.einsum(
        #             'vh,bsh->bsv', [self.output_embeddings.weight, f]
        #         )
        #         return logits
        #     else:
        #         if self.config.predict_source:
        #             field_ids = input_ids[const.SOURCE]
        #         else:
        #             field_ids = input_ids[const.TARGET]
        #         if isinstance(field_ids, tuple):
        #             raise DeprecationWarning
        #             # field_ids, _field_lengths = field_ids
        #         elif isinstance(field_ids, BatchedSentence):
        #             field_ids = field_ids.tensor
        #
        #         PreQEFV = torch.einsum(
        #             'bsh,bsh->bsh', [self.output_embeddings(field_ids[:, 1:-1]), f]
        #         )
        #         start = self.start_PreQEFV.expand(PreQEFV.size(0), -1, -1)
        #         end = self.end_PreQEFV.expand(PreQEFV.size(0), -1, -1)
        #         PreQEFV = torch.cat((start, PreQEFV, end), dim=1)
        #
        #         PostQEFV = torch.cat([forward_contexts, backward_contexts], dim=-1)
        #
        #         features = torch.cat([PreQEFV, PostQEFV], dim=-1)
        #         return features

        # if use_mismatch_features:
        #     logits = torch.einsum('vh,bsh->bsv', [self.output_embeddings.weight, f])
        #
        #     # get target and prediction ids (this target doesnt mean mt)
        #     target = field_ids[:, 1:-1]
        #     pred = torch.argmax(logits, dim=-1)
        #
        #     # calculate mismatch features and concat them
        #     t_max = torch.gather(logits, -1, target.unsqueeze(-1))
        #     p_max = torch.gather(logits, -1, pred.unsqueeze(-1))
        #     diff_max = t_max - p_max
        #     diff_arg = (target != pred).float().unsqueeze(-1)
        #     mismatch = torch.cat((t_max, p_max, diff_max, diff_arg), dim=-1)
        #
        #     # add a start and end token to mismatch features
        #     start = torch.zeros(
        #         (mismatch.size(0), 1, mismatch.size(-1)), device=mismatch.device
        #     )
        #     end = torch.zeros(
        #         (mismatch.size(0), 1, mismatch.size(-1)), device=mismatch.device
        #     )
        #     mismatch = torch.cat((start, mismatch, end), dim=1)
        #
        #     # concat mismatch features with predictor features
        #     features = torch.cat((features, mismatch), dim=-1)

        return output_features
