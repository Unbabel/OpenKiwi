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
from collections import OrderedDict

import torch
from torch import nn
from torch.distributions.normal import Normal

from kiwi import constants as const
from kiwi.metrics import (
    CorrectMetric,
    ExpectedErrorMetric,
    F1Metric,
    LogMetric,
    PearsonMetric,
    PerplexityMetric,
    RMSEMetric,
    SpearmanMetric,
    ThresholdCalibrationMetric,
    TokenMetric,
)
from kiwi.models.model import Model
from kiwi.models.predictor import Predictor, PredictorConfig
from kiwi.models.utils import apply_packed_sequence, make_loss_weights

logger = logging.getLogger(__name__)


class EstimatorConfig(PredictorConfig):
    def __init__(
        self,
        vocabs,
        hidden_est=100,
        rnn_layers_est=1,
        mlp_est=True,
        dropout_est=0.0,
        start_stop=False,
        predict_target=True,
        predict_gaps=False,
        predict_source=False,
        token_level=True,
        sentence_level=True,
        sentence_ll=True,
        binary_level=True,
        target_bad_weight=2.0,
        source_bad_weight=2.0,
        gaps_bad_weight=2.0,
        **kwargs
    ):
        """Predictor Estimator Hyperparams.
        """
        super().__init__(vocabs, **kwargs)
        self.start_stop = start_stop or predict_gaps
        self.hidden_est = hidden_est
        self.rnn_layers_est = rnn_layers_est
        self.mlp_est = mlp_est
        self.dropout_est = dropout_est
        self.predict_target = predict_target
        self.predict_gaps = predict_gaps
        self.predict_source = predict_source
        self.token_level = token_level
        self.sentence_level = sentence_level
        self.sentence_ll = sentence_ll
        self.binary_level = binary_level
        self.target_bad_weight = target_bad_weight
        self.source_bad_weight = source_bad_weight
        self.gaps_bad_weight = gaps_bad_weight


@Model.register_subclass
class Estimator(Model):
    title = 'PredEst (Predictor-Estimator)'

    def __init__(
        self, vocabs, predictor_tgt=None, predictor_src=None, **kwargs
    ):

        super().__init__(vocabs=vocabs, ConfigCls=EstimatorConfig, **kwargs)

        if predictor_src:
            self.config.update(predictor_src.config)
        elif predictor_tgt:
            self.config.update(predictor_tgt.config)

        # Predictor Settings #
        predict_tgt = (
            self.config.predict_target
            or self.config.predict_gaps
            or self.config.sentence_level
        )
        if predict_tgt and not predictor_tgt:
            predictor_tgt = Predictor(
                vocabs=vocabs,
                predict_inverse=False,
                hidden_pred=self.config.hidden_pred,
                rnn_layers_pred=self.config.rnn_layers_pred,
                dropout_pred=self.config.dropout_pred,
                target_embeddings_size=self.config.target_embeddings_size,
                source_embeddings_size=self.config.source_embeddings_size,
                out_embeddings_size=self.config.out_embeddings_size,
            )
        if self.config.predict_source and not predictor_src:
            predictor_src = Predictor(
                vocabs=vocabs,
                predict_inverse=True,
                hidden_pred=self.config.hidden_pred,
                rnn_layers_pred=self.config.rnn_layers_pred,
                dropout_pred=self.config.dropout_pred,
                target_embeddings_size=self.config.target_embeddings_size,
                source_embeddings_size=self.config.source_embeddings_size,
                out_embeddings_size=self.config.out_embeddings_size,
            )

        # Update the predictor vocabs if token level == True
        # Required by `get_mask` call in predictor forward with `pe` side
        # to determine padding IDs.
        if self.config.token_level:
            if predictor_src:
                predictor_src.vocabs = vocabs
            if predictor_tgt:
                predictor_tgt.vocabs = vocabs

        self.predictor_tgt = predictor_tgt
        self.predictor_src = predictor_src

        predictor_hidden = self.config.hidden_pred
        embedding_size = self.config.out_embeddings_size
        input_size = 2 * predictor_hidden + embedding_size

        self.nb_classes = len(const.LABELS)
        self.lstm_input_size = input_size

        self.mlp = None
        self.sentence_pred = None
        self.sentence_sigma = None
        self.binary_pred = None
        self.binary_scale = None

        # Build Model #

        if self.config.start_stop:
            self.start_PreQEFV = nn.Parameter(torch.zeros(1, 1, embedding_size))
            self.end_PreQEFV = nn.Parameter(torch.zeros(1, 1, embedding_size))

        if self.config.mlp_est:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, self.config.hidden_est), nn.Tanh()
            )
            self.lstm_input_size = self.config.hidden_est

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.config.hidden_est,
            num_layers=self.config.rnn_layers_est,
            batch_first=True,
            dropout=self.config.dropout_est,
            bidirectional=True,
        )
        self.embedding_out = nn.Linear(
            2 * self.config.hidden_est, self.nb_classes
        )
        if self.config.predict_gaps:
            self.embedding_out_gaps = nn.Linear(
                4 * self.config.hidden_est, self.nb_classes
            )
        self.dropout = None
        if self.config.dropout_est:
            self.dropout = nn.Dropout(self.config.dropout_est)

        # Multitask Learning Objectives #
        sentence_input_size = (
            2 * self.config.rnn_layers_est * self.config.hidden_est
        )
        if self.config.sentence_level:
            self.sentence_pred = nn.Sequential(
                nn.Linear(sentence_input_size, sentence_input_size // 2),
                nn.Sigmoid(),
                nn.Linear(sentence_input_size // 2, sentence_input_size // 4),
                nn.Sigmoid(),
                nn.Linear(sentence_input_size // 4, 1),
            )
            self.sentence_sigma = None
            if self.config.sentence_ll:
                # Predict truncated Gaussian distribution
                self.sentence_sigma = nn.Sequential(
                    nn.Linear(sentence_input_size, sentence_input_size // 2),
                    nn.Sigmoid(),
                    nn.Linear(
                        sentence_input_size // 2, sentence_input_size // 4
                    ),
                    nn.Sigmoid(),
                    nn.Linear(sentence_input_size // 4, 1),
                    nn.Sigmoid(),
                )
        if self.config.binary_level:
            self.binary_pred = nn.Sequential(
                nn.Linear(sentence_input_size, sentence_input_size // 2),
                nn.Tanh(),
                nn.Linear(sentence_input_size // 2, sentence_input_size // 4),
                nn.Tanh(),
                nn.Linear(sentence_input_size // 4, 2),
            )

        # Build Losses #

        # FIXME: Remove dependency on magic numbers
        self.xents = nn.ModuleDict()
        weight = make_loss_weights(
            self.nb_classes, const.BAD_ID, self.config.target_bad_weight
        )

        self.xents[const.TARGET_TAGS] = nn.CrossEntropyLoss(
            reduction='sum', ignore_index=const.PAD_TAGS_ID, weight=weight
        )
        if self.config.predict_source:
            weight = make_loss_weights(
                self.nb_classes, const.BAD_ID, self.config.source_bad_weight
            )
            self.xents[const.SOURCE_TAGS] = nn.CrossEntropyLoss(
                reduction='sum', ignore_index=const.PAD_TAGS_ID, weight=weight
            )
        if self.config.predict_gaps:
            weight = make_loss_weights(
                self.nb_classes, const.BAD_ID, self.config.gaps_bad_weight
            )
            self.xents[const.GAP_TAGS] = nn.CrossEntropyLoss(
                reduction='sum', ignore_index=const.PAD_TAGS_ID, weight=weight
            )
        if self.config.sentence_level and not self.config.sentence_ll:
            self.mse_loss = nn.MSELoss(reduction='sum')
        if self.config.binary_level:
            self.xent_binary = nn.CrossEntropyLoss(reduction='sum')

    @staticmethod
    def fieldset(*args, **kwargs):
        from kiwi.data.fieldsets.predictor_estimator import build_fieldset

        return build_fieldset(*args, **kwargs)

    @staticmethod
    def from_options(vocabs, opts):
        """

        Args:
            vocabs:
            opts:
                predict_target (bool): Predict target tags
                predict_source (bool): Predict source tags
                predict_gaps (bool): Predict gap tags
                token_level (bool): Train predictor using PE field.
                sentence_level (bool): Predict Sentence Scores
                sentence_ll (bool): Use likelihood loss for sentence scores
                                    (instead of squared error)
                binary_level: Predict binary sentence labels
                target_bad_weight: Weight for target tags bad class. Default 3.0
                source_bad_weight: Weight for source tags bad class. Default 3.0
                gaps_bad_weight: Weight for gap tags bad class. Default 3.0

        Returns:

        """
        predictor_src = predictor_tgt = None
        if opts.load_pred_source:
            predictor_src = Predictor.from_file(opts.load_pred_source)
        if opts.load_pred_target:
            predictor_tgt = Predictor.from_file(opts.load_pred_target)

        model = Estimator(
            vocabs,
            predictor_tgt=predictor_tgt,
            predictor_src=predictor_src,
            hidden_est=opts.hidden_est,
            rnn_layers_est=opts.rnn_layers_est,
            mlp_est=opts.mlp_est,
            dropout_est=opts.dropout_est,
            start_stop=opts.start_stop,
            predict_target=opts.predict_target,
            predict_gaps=opts.predict_gaps,
            predict_source=opts.predict_source,
            token_level=opts.token_level,
            sentence_level=opts.sentence_level,
            sentence_ll=opts.sentence_ll,
            binary_level=opts.binary_level,
            target_bad_weight=opts.target_bad_weight,
            source_bad_weight=opts.source_bad_weight,
            gaps_bad_weight=opts.gaps_bad_weight,
            hidden_pred=opts.hidden_pred,
            rnn_layers_pred=opts.rnn_layers_pred,
            dropout_pred=opts.dropout_pred,
            share_embeddings=opts.dropout_est,
            embedding_sizes=opts.embedding_sizes,
            target_embeddings_size=opts.target_embeddings_size,
            source_embeddings_size=opts.source_embeddings_size,
            out_embeddings_size=opts.out_embeddings_size,
            predict_inverse=opts.predict_inverse,
        )
        return model

    def forward(self, batch):
        outputs = OrderedDict()
        contexts_tgt, h_tgt = None, None
        contexts_src, h_src = None, None
        if (
            self.config.predict_target
            or self.config.predict_gaps
            or self.config.sentence_level
        ):
            model_out_tgt = self.predictor_tgt(batch)
            input_seq, target_lengths = self.make_input(
                model_out_tgt, batch, const.TARGET_TAGS
            )

            contexts_tgt, h_tgt = apply_packed_sequence(
                self.lstm, input_seq, target_lengths
            )
            if self.config.predict_target:
                logits = self.predict_tags(contexts_tgt)
                if self.config.start_stop:
                    logits = logits[:, 1:-1]
                outputs[const.TARGET_TAGS] = logits

            if self.config.predict_gaps:
                contexts_gaps = self.make_contexts_gaps(contexts_tgt)
                logits = self.predict_tags(
                    contexts_gaps, out_embed=self.embedding_out_gaps
                )
                outputs[const.GAP_TAGS] = logits
        if self.config.predict_source:
            model_out_src = self.predictor_src(batch)
            input_seq, target_lengths = self.make_input(
                model_out_src, batch, const.SOURCE_TAGS
            )
            contexts_src, h_src = apply_packed_sequence(
                self.lstm, input_seq, target_lengths
            )

            logits = self.predict_tags(contexts_src)
            outputs[const.SOURCE_TAGS] = logits

        # Sentence/Binary/Token Level prediction
        sentence_input = self.make_sentence_input(h_tgt, h_src)
        if self.config.sentence_level:
            outputs.update(self.predict_sentence(sentence_input))

        if self.config.binary_level:
            bin_logits = self.binary_pred(sentence_input).squeeze()
            outputs[const.BINARY] = bin_logits

        if self.config.token_level and hasattr(batch, const.PE):
            if self.predictor_tgt:
                model_out = self.predictor_tgt(batch, target_side=const.PE)
                logits = model_out[const.PE]
                outputs[const.PE] = logits
            if self.predictor_src:
                model_out = self.predictor_src(batch, source_side=const.PE)
                logits = model_out[const.SOURCE]
                outputs[const.SOURCE] = logits

        # TODO remove?
        # if self.use_probs:
        #     logits -= logits.mean(-1, keepdim=True)
        #     logits_exp = logits.exp()
        #     logprobs = logits - logits_exp.sum(-1, keepdim=True).log()
        #     sentence_scores = ((logprobs.exp() * token_mask).sum(1)
        #                        / target_lengths)
        #     sentence_scores = sentence_scores[..., 1 - self.BAD_ID]
        #     binary_logits = (logprobs * token_mask).sum(1)

        return outputs

    def make_input(self, model_out, batch, tagset):
        """Make Input Sequence from predictor outputs. """
        PreQEFV = model_out[const.PREQEFV]
        PostQEFV = model_out[const.POSTQEFV]
        side = const.TARGET
        if tagset == const.SOURCE_TAGS:
            side = const.SOURCE
        token_mask = self.get_mask(batch, side)
        batch_size = token_mask.shape[0]
        target_lengths = token_mask.sum(1)
        if self.config.start_stop:
            target_lengths += 2
            start = self.start_PreQEFV.expand(
                batch_size, 1, self.config.out_embeddings_size
            )
            end = self.end_PreQEFV.expand(
                batch_size, 1, self.config.out_embeddings_size
            )
            PreQEFV = torch.cat((start, PreQEFV, end), dim=1)
        else:
            PostQEFV = PostQEFV[:, 1:-1]

        input_seq = torch.cat([PreQEFV, PostQEFV], dim=-1)
        length, input_dim = input_seq.shape[1:]
        if self.mlp:
            input_flat = input_seq.view(batch_size * length, input_dim)
            input_flat = self.mlp(input_flat)
            input_seq = input_flat.view(
                batch_size, length, self.lstm_input_size
            )
        return input_seq, target_lengths

    def make_contexts_gaps(self, contexts):
        # Concat Contexts Shifted
        contexts = torch.cat((contexts[:, :-1], contexts[:, 1:]), dim=-1)
        return contexts

    def make_sentence_input(self, h_tgt, h_src):
        """Reshape last hidden state. """
        h = h_tgt[0] if h_tgt else h_src[0]
        h = h.contiguous().transpose(0, 1)
        return h.reshape(h.shape[0], -1)

    def predict_sentence(self, sentence_input):
        """Compute Sentence Score predictions."""
        outputs = OrderedDict()
        sentence_scores = self.sentence_pred(sentence_input).squeeze()
        outputs[const.SENTENCE_SCORES] = sentence_scores
        if self.sentence_sigma:
            # Predict truncated Gaussian on [0,1]
            sigma = self.sentence_sigma(sentence_input).squeeze()
            outputs[const.SENT_SIGMA] = sigma
            outputs['SENT_MU'] = outputs[const.SENTENCE_SCORES]
            mean = outputs['SENT_MU'].clone().detach()
            # Compute log-likelihood of x given mu, sigma
            normal = Normal(mean, sigma)
            # Renormalize on [0,1] for truncated Gaussian
            partition_function = (normal.cdf(1) - normal.cdf(0)).detach()
            outputs[const.SENTENCE_SCORES] = mean + (
                (
                    sigma ** 2
                    * (normal.log_prob(0).exp() - normal.log_prob(1).exp())
                )
                / partition_function
            )

        return outputs

    def predict_tags(self, contexts, out_embed=None):
        """Compute Tag Predictions."""
        if not out_embed:
            out_embed = self.embedding_out
        batch_size, length, hidden = contexts.shape
        if self.dropout:
            contexts = self.dropout(contexts)
        # Fold sequence length in batch dimension
        contexts_flat = contexts.contiguous().view(-1, hidden)
        logits_flat = out_embed(contexts_flat)
        logits = logits_flat.view(batch_size, length, self.nb_classes)
        return logits

    def sentence_loss(self, model_out, batch):
        """Compute Sentence score loss"""
        sentence_pred = model_out[const.SENTENCE_SCORES]
        sentence_scores = batch.sentence_scores
        if not self.sentence_sigma:
            return self.mse_loss(sentence_pred, sentence_scores)
        else:
            sigma = model_out[const.SENT_SIGMA]
            mean = model_out['SENT_MU']
            # Compute log-likelihood of x given mu, sigma
            normal = Normal(mean, sigma)
            # Renormalize on [0,1] for truncated Gaussian
            partition_function = (normal.cdf(1) - normal.cdf(0)).detach()
            nll = partition_function.log() - normal.log_prob(sentence_scores)
            return nll.sum()

    def word_loss(self, model_out, batch):
        """Compute Sequence Tagging Loss"""
        word_loss = OrderedDict()
        for tag in const.TAGS:
            if tag in model_out:
                logits = model_out[tag]
                logits = logits.transpose(1, 2)
                word_loss[tag] = self.xents[tag](logits, getattr(batch, tag))
        return word_loss

    def binary_loss(self, model_out, batch):
        """Compute Sentence Classification Loss"""
        labels = getattr(batch, const.BINARY)
        loss = self.xent_binary(model_out[const.BINARY], labels.long())
        return loss

    def loss(self, model_out, batch):
        """Compute Model Loss"""
        loss_dict = self.word_loss(model_out, batch)
        if self.config.sentence_level:
            loss_sent = self.sentence_loss(model_out, batch)
            loss_dict[const.SENTENCE_SCORES] = loss_sent
        if self.config.binary_level:
            loss_bin = self.binary_loss(model_out, batch)
            loss_dict[const.BINARY] = loss_bin

        if const.PE in model_out:
            loss_token = self.predictor_tgt.loss(
                model_out, batch, target_side=const.PE
            )
            loss_dict[const.PE] = loss_token[const.PE]
        if const.SOURCE in model_out:
            loss_token = self.predictor_src.loss(model_out, batch)
            loss_dict[const.SOURCE] = loss_token[const.SOURCE]

        loss_dict[const.LOSS] = sum(loss.sum() for _, loss in loss_dict.items())
        return loss_dict

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
                ThresholdCalibrationMetric(
                    prefix=const.TARGET_TAGS,
                    target_name=const.TARGET_TAGS,
                    PAD=const.PAD_TAGS_ID,
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

        if self.config.sentence_level:
            metrics.append(RMSEMetric(target_name=const.SENTENCE_SCORES))
            metrics.append(PearsonMetric(target_name=const.SENTENCE_SCORES))
            metrics.append(SpearmanMetric(target_name=const.SENTENCE_SCORES))
            if self.config.sentence_ll:
                metrics.append(
                    LogMetric(targets=[('model_out', const.SENT_SIGMA)])
                )
        if self.config.binary_level:
            metrics.append(
                CorrectMetric(prefix=const.BINARY, target_name=const.BINARY)
            )
        if self.config.token_level and self.predictor_tgt is not None:
            metrics.append(
                CorrectMetric(
                    prefix=const.PE,
                    target_name=const.PE,
                    PAD=const.PAD_ID,
                    STOP=const.STOP_ID,
                )
            )
            metrics.append(
                ExpectedErrorMetric(
                    prefix=const.PE,
                    target_name=const.PE,
                    PAD=const.PAD_ID,
                    STOP=const.STOP_ID,
                )
            )
            metrics.append(
                PerplexityMetric(
                    prefix=const.PE,
                    target_name=const.PE,
                    PAD=const.PAD_ID,
                    STOP=const.STOP_ID,
                )
            )
        if self.config.token_level and self.predictor_src is not None:
            metrics.append(
                CorrectMetric(
                    prefix=const.SOURCE,
                    target_name=const.SOURCE,
                    PAD=const.PAD_ID,
                    STOP=const.STOP_ID,
                )
            )
            metrics.append(
                ExpectedErrorMetric(
                    prefix=const.SOURCE,
                    target_name=const.SOURCE,
                    PAD=const.PAD_ID,
                    STOP=const.STOP_ID,
                )
            )
            metrics.append(
                PerplexityMetric(
                    prefix=const.SOURCE,
                    target_name=const.SOURCE,
                    PAD=const.PAD_ID,
                    STOP=const.STOP_ID,
                )
            )
        metrics.append(
            TokenMetric(
                target_name=const.TARGET, STOP=const.STOP_ID, PAD=const.PAD_ID
            )
        )
        return metrics

    def metrics_ordering(self):
        return max
