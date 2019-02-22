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
from torch import nn

from kiwi import constants as const
from kiwi.metrics import CorrectMetric, ExpectedErrorMetric, PerplexityMetric
from kiwi.models.model import Model, ModelConfig
from kiwi.models.modules.attention import Attention
from kiwi.models.modules.scorer import MLPScorer
from kiwi.models.utils import apply_packed_sequence, replace_token


class PredictorConfig(ModelConfig):
    def __init__(
        self,
        vocabs,
        hidden_pred=400,
        rnn_layers_pred=3,
        dropout_pred=0.0,
        share_embeddings=False,
        embedding_sizes=0,
        target_embeddings_size=200,
        source_embeddings_size=200,
        out_embeddings_size=200,
        predict_inverse=False,
    ):
        """Predictor Hyperparams.
        """
        super().__init__(vocabs)

        # Vocabulary
        self.target_side = const.TARGET
        self.source_side = const.SOURCE
        self.predict_inverse = predict_inverse
        if self.predict_inverse:
            self.source_side, self.target_side = (
                self.target_side,
                self.source_side,
            )
            self.target_vocab_size, self.source_vocab_size = (
                self.source_vocab_size,
                self.target_vocab_size,
            )

        # Architecture
        self.hidden_pred = hidden_pred
        self.rnn_layers_pred = rnn_layers_pred
        self.dropout_pred = dropout_pred
        self.share_embeddings = share_embeddings
        if embedding_sizes:
            self.target_embeddings_size = embedding_sizes
            self.source_embeddings_size = embedding_sizes
            self.out_embeddings_size = embedding_sizes
        else:
            self.target_embeddings_size = target_embeddings_size
            self.source_embeddings_size = source_embeddings_size
            self.out_embeddings_size = out_embeddings_size


@Model.register_subclass
class Predictor(Model):
    """Bidirectional Conditional Language Model

       Implemented after Kim et al 2017, see:
         http://www.statmt.org/wmt17/pdf/WMT63.pdf
    """

    title = 'PredEst Predictor model (an embedder model)'

    def __init__(self, vocabs, **kwargs):
        """
        Args:
          vocabs: Dictionary Mapping Field Names to Vocabularies.
        kwargs:
          config: A state dict of a PredictorConfig object.
          dropout: LSTM dropout Default 0.0
          hidden_pred: LSTM Hidden Size, default 200
          rnn_layers: Default 3
          embedding_sizes: If set, takes precedence over other embedding params
                           Default 100
          source_embeddings_size: Default 100
          target_embeddings_size: Default 100
          out_embeddings_size: Output softmax embedding. Default 100
          share_embeddings: Tie input and output embeddings for target.
                            Default False
          predict_inverse: Predict from target to source. Default False
        """
        super().__init__(vocabs=vocabs, ConfigCls=PredictorConfig, **kwargs)

        scorer = MLPScorer(
            self.config.hidden_pred * 2, self.config.hidden_pred * 2, layers=2
        )

        self.attention = Attention(scorer)
        self.embedding_source = nn.Embedding(
            self.config.source_vocab_size,
            self.config.source_embeddings_size,
            const.PAD_ID,
        )
        self.embedding_target = nn.Embedding(
            self.config.target_vocab_size,
            self.config.target_embeddings_size,
            const.PAD_ID,
        )
        self.lstm_source = nn.LSTM(
            input_size=self.config.source_embeddings_size,
            hidden_size=self.config.hidden_pred,
            num_layers=self.config.rnn_layers_pred,
            batch_first=True,
            dropout=self.config.dropout_pred,
            bidirectional=True,
        )
        self.forward_target = nn.LSTM(
            input_size=self.config.target_embeddings_size,
            hidden_size=self.config.hidden_pred,
            num_layers=self.config.rnn_layers_pred,
            batch_first=True,
            dropout=self.config.dropout_pred,
            bidirectional=False,
        )
        self.backward_target = nn.LSTM(
            input_size=self.config.target_embeddings_size,
            hidden_size=self.config.hidden_pred,
            num_layers=self.config.rnn_layers_pred,
            batch_first=True,
            dropout=self.config.dropout_pred,
            bidirectional=False,
        )

        self.W1 = self.embedding_target
        if not self.config.share_embeddings:
            self.W1 = nn.Embedding(
                self.config.target_vocab_size,
                self.config.out_embeddings_size,
                const.PAD_ID,
            )
        self.W2 = nn.Parameter(
            torch.zeros(
                self.config.out_embeddings_size, self.config.out_embeddings_size
            )
        )
        self.V = nn.Parameter(
            torch.zeros(
                2 * self.config.target_embeddings_size,
                2 * self.config.out_embeddings_size,
            )
        )
        self.C = nn.Parameter(
            torch.zeros(
                2 * self.config.hidden_pred, 2 * self.config.out_embeddings_size
            )
        )
        self.S = nn.Parameter(
            torch.zeros(
                2 * self.config.hidden_pred, 2 * self.config.out_embeddings_size
            )
        )
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        self._loss = nn.CrossEntropyLoss(
            reduction='sum', ignore_index=const.PAD_ID
        )

    @staticmethod
    def fieldset(*args, **kwargs):
        from kiwi.data.fieldsets.predictor import build_fieldset

        return build_fieldset()

    @staticmethod
    def from_options(vocabs, opts):
        """

        Args:
            vocabs:
            opts:

        Returns:

        """
        model = Predictor(
            vocabs,
            hidden_pred=opts.hidden_pred,
            rnn_layers_pred=opts.rnn_layers_pred,
            dropout_pred=opts.dropout_pred,
            share_embeddings=opts.share_embeddings,
            embedding_sizes=opts.embedding_sizes,
            target_embeddings_size=opts.target_embeddings_size,
            source_embeddings_size=opts.source_embeddings_size,
            out_embeddings_size=opts.out_embeddings_size,
            predict_inverse=opts.predict_inverse,
        )
        return model

    def loss(self, model_out, batch, target_side=None):
        if not target_side:
            target_side = self.config.target_side
        target = getattr(batch, target_side)
        # There are no predictions for first/last element
        target = replace_token(target[:, 1:-1], const.STOP_ID, const.PAD_ID)
        # Predicted Class must be in dim 1 for xentropyloss
        logits = model_out[target_side]
        logits = logits.transpose(1, 2)
        loss = self._loss(logits, target)
        loss_dict = OrderedDict()
        loss_dict[target_side] = loss
        loss_dict[const.LOSS] = loss
        return loss_dict

    def forward(self, batch, source_side=None, target_side=None):
        if not source_side:
            source_side = self.config.source_side
        if not target_side:
            target_side = self.config.target_side

        source = getattr(batch, source_side)
        target = getattr(batch, target_side)
        batch_size, target_len = target.shape[:2]
        # Remove First and Last Element (Start / Stop Tokens)
        source_mask = self.get_mask(batch, source_side)[:, 1:-1]
        source_lengths = source_mask.sum(1)
        target_lengths = self.get_mask(batch, target_side).sum(1)
        source_embeddings = self.embedding_source(source)
        target_embeddings = self.embedding_target(target)
        # Source Encoding
        source_contexts, hidden = apply_packed_sequence(
            self.lstm_source, source_embeddings, source_lengths
        )
        # Target Encoding.
        h_forward, h_backward = self._split_hidden(hidden)
        forward_contexts, _ = self.forward_target(target_embeddings, h_forward)
        target_emb_rev = self._reverse_padded_seq(
            target_lengths, target_embeddings
        )
        backward_contexts, _ = self.backward_target(target_emb_rev, h_backward)
        backward_contexts = self._reverse_padded_seq(
            target_lengths, backward_contexts
        )

        # For each position, concatenate left context i-1 and right context i+1
        target_contexts = torch.cat(
            [forward_contexts[:, :-2], backward_contexts[:, 2:]], dim=-1
        )
        # For each position i, concatenate Emeddings i-1 and i+1
        target_embeddings = torch.cat(
            [target_embeddings[:, :-2], target_embeddings[:, 2:]], dim=-1
        )

        # Get Attention vectors for all positions and stack.
        self.attention.set_mask(source_mask.float())
        attns = [
            self.attention(
                target_contexts[:, i], source_contexts, source_contexts
            )
            for i in range(target_len - 2)
        ]
        attns = torch.stack(attns, dim=1)

        # Combine attention, embeddings and target context vectors
        C = torch.einsum('bsi,il->bsl', [attns, self.C])
        V = torch.einsum('bsj,jl->bsl', [target_embeddings, self.V])
        S = torch.einsum('bsk,kl->bsl', [target_contexts, self.S])
        t_tilde = C + V + S
        # Maxout with pooling size 2
        t, _ = torch.max(
            t_tilde.view(
                t_tilde.shape[0], t_tilde.shape[1], t_tilde.shape[-1] // 2, 2
            ),
            dim=-1,
        )

        f = torch.einsum('oh,bso->bsh', [self.W2, t])
        logits = torch.einsum('vh,bsh->bsv', [self.W1.weight, f])
        PreQEFV = torch.einsum('bsh,bsh->bsh', [self.W1(target[:, 1:-1]), f])
        PostQEFV = torch.cat([forward_contexts, backward_contexts], dim=-1)
        return {
            target_side: logits,
            const.PREQEFV: PreQEFV,
            const.POSTQEFV: PostQEFV,
        }

    @staticmethod
    def _reverse_padded_seq(lengths, sequence):
        """ Reverses a batch of padded sequences of different length.
        """
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
        """Split Hidden State into forward/backward parts.
        """
        h, c = hidden
        size = h.shape[0]
        idx_forward = torch.arange(0, size, 2, dtype=torch.long)
        idx_backward = torch.arange(1, size, 2, dtype=torch.long)
        hidden_forward = (h[idx_forward], c[idx_forward])
        hidden_backward = (h[idx_backward], c[idx_backward])
        return hidden_forward, hidden_backward

    def metrics(self):
        metrics = []

        main_metric = PerplexityMetric(
            prefix=self.config.target_side,
            target_name=self.config.target_side,
            PAD=const.PAD_ID,
            STOP=const.STOP_ID,
        )
        metrics.append(main_metric)

        metrics.append(
            CorrectMetric(
                prefix=self.config.target_side,
                target_name=self.config.target_side,
                PAD=const.PAD_ID,
                STOP=const.STOP_ID,
            )
        )
        metrics.append(
            ExpectedErrorMetric(
                prefix=self.config.target_side,
                target_name=self.config.target_side,
                PAD=const.PAD_ID,
                STOP=const.STOP_ID,
            )
        )
        return metrics

    def metrics_ordering(self):
        return min
