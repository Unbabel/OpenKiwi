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
)
from kiwi.models.model import Model, QEModelConfig
from kiwi.models.predictor import Predictor, PredictorConfig
from kiwi.models.utils import apply_packed_sequence, make_loss_weights

logger = logging.getLogger(__name__)


class EstimatorConfig(PredictorConfig, QEModelConfig):
    def __init__(
        self,
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
        super().__init__(
            predict_target=predict_target,
            predict_source=predict_source,
            predict_gaps=predict_gaps,
            **kwargs
        )
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
        self.predictors = nn.ModuleDict()
        predict_tgt = self.config.predict_target or self.config.predict_gaps
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
            self.predictors[const.TARGET] = predictor_tgt
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
            self.predictors[const.SOURCE] = predictor_src

        # Update the predictor vocabs if token level == True
        # Required by `get_mask` call in predictor forward with `pe` side
        # to determine padding IDs.
        if self.config.token_level:
            if predictor_src:
                predictor_src.vocabs = vocabs
            if predictor_tgt:
                predictor_tgt.vocabs = vocabs

        predictor_hidden = self.config.hidden_pred
        embedding_size = self.config.out_embeddings_size
        input_size = 2 * predictor_hidden + embedding_size

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
        self.dropout = None
        if self.config.dropout_est:
            self.dropout = nn.Dropout(self.config.dropout_est)

        # Word Level Objectives #
        self.xents = nn.ModuleDict()
        self.tags_output_emb = nn.ModuleDict()
        for tag in self.config.output_tags:
            if tag == const.GAP_TAGS:
                tag_embedding_size = 4 * self.config.hidden_est
            else:
                tag_embedding_size = 2 * self.config.hidden_est
            self.tags_output_emb[tag] = nn.Linear(
                tag_embedding_size,
                self.config.nb_classes[tag]
            )
            weight = make_loss_weights(
                self.config.nb_classes[tag],
                self.config.bad_idx[tag],
                self.config.bad_weights[tag]
            )
            self.xents[tag] = nn.CrossEntropyLoss(
                reduction='sum',
                ignore_index=self.config.pad_idx[tag],
                weight=weight
            )

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
        contexts, hs = {}, {}
        for tag in self.config.output_tags:
            if tag == const.SOURCE_TAGS:
                side = const.SOURCE
            else:
                side = const.TARGET
            if side not in contexts:
                model_out = self.predictors[side](batch)
                input_seq, target_lengths = self.make_input(
                    model_out, batch, side
                )
                contexts[side], hs[side] = apply_packed_sequence(
                    self.lstm, input_seq, target_lengths
                )
            logits = self.predict_tags(contexts[side], tag=tag)
            outputs[tag] = logits

        # Sentence/Binary/Token Level prediction
        sentence_input = self.make_sentence_input(hs)
        if self.config.sentence_level:
            outputs.update(self.predict_sentence(sentence_input))

        if self.config.binary_level:
            bin_logits = self.binary_pred(sentence_input).squeeze()
            outputs[const.BINARY] = bin_logits

        if self.config.token_level and hasattr(batch, const.PE):
            if const.TARGET in self.predictors:
                model_out = self.predictors[const.TARGET](
                    batch, target_side=const.PE
                )
                logits = model_out[const.PE]
                outputs[const.PE] = logits
            if const.SOURCE in self.predictors:
                model_out = self.predictors[const.SOURCE](
                    batch, source_side=const.PE
                )
                logits = model_out[const.SOURCE]
                outputs[const.SOURCE] = logits

        return outputs

    def make_input(self, model_out, batch, side):
        """Make Input Sequence from predictor outputs. """
        PreQEFV = model_out[const.PREQEFV]
        PostQEFV = model_out[const.POSTQEFV]
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

    def make_sentence_input(self, h):
        """Reshape last hidden state. """
        h = h[const.TARGET][0] if const.TARGET in h else h[const.SOURCE][0]
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
            # Predict the mean of truncated Gaussian with shape sigma, mu
            partition_function = (normal.cdf(1) - normal.cdf(0)).detach()
            outputs[const.SENTENCE_SCORES] = mean + (
                (
                    sigma ** 2
                    * (normal.log_prob(0).exp() - normal.log_prob(1).exp())
                )
                / partition_function
            )

        return outputs

    def predict_tags(self, contexts, tag):
        """Compute Tag Predictions."""
        if tag == const.GAP_TAGS:
            contexts = self.make_contexts_gaps(contexts)
        batch_size, length, hidden = contexts.shape
        if self.dropout:
            contexts = self.dropout(contexts)
        # Fold sequence length in batch dimension
        contexts_flat = contexts.contiguous().view(-1, hidden)
        logits_flat = self.tags_output_emb[tag](contexts_flat)
        logits = logits_flat.view(
            batch_size, length, self.config.nb_classes[tag]
        )
        if self.config.start_stop and tag != const.GAP_TAGS:
            logits = logits[:, 1:-1]
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
        for tag in self.config.output_tags:
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
            loss_token = self.predictors[const.TARGET].loss(
                model_out, batch, target_side=const.PE
            )
            loss_dict[const.PE] = loss_token[const.PE]
        if const.SOURCE in model_out:
            loss_token = self.predictors[const.SOURCE].loss(
                model_out, batch
            )
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
                    PAD=self.config.pad_idx[const.TARGET_TAGS],
                    labels=const.LABELS,
                )
            )
            metrics.append(
                ThresholdCalibrationMetric(
                    prefix=const.TARGET_TAGS,
                    target_name=const.TARGET_TAGS,
                    PAD=self.config.pad_idx[const.TARGET_TAGS],
                )
            )
            metrics.append(
                CorrectMetric(
                    prefix=const.TARGET_TAGS,
                    target_name=const.TARGET_TAGS,
                    PAD=self.config.pad_idx[const.TARGET_TAGS],
                )
            )

        if self.config.predict_source:
            metrics.append(
                F1Metric(
                    prefix=const.SOURCE_TAGS,
                    target_name=const.SOURCE_TAGS,
                    PAD=self.config.pad_idx[const.SOURCE_TAGS],
                    labels=const.LABELS,
                )
            )
            metrics.append(
                CorrectMetric(
                    prefix=const.SOURCE_TAGS,
                    target_name=const.SOURCE_TAGS,
                    PAD=self.config.pad_idx[const.SOURCE_TAGS],
                )
            )
        if self.config.predict_gaps:
            metrics.append(
                F1Metric(
                    prefix=const.GAP_TAGS,
                    target_name=const.GAP_TAGS,
                    PAD=self.config.pad_idx[const.GAP_TAGS],
                    labels=const.LABELS,
                )
            )
            metrics.append(
                CorrectMetric(
                    prefix=const.GAP_TAGS,
                    target_name=const.GAP_TAGS,
                    PAD=self.config.pad_idx[const.GAP_TAGS],
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

        if self.config.token_level and const.TARGET in self.predictors:
            metrics.append(
                CorrectMetric(
                    prefix=const.PE,
                    target_name=const.PE,
                    PAD=self.config.pad_idx[const.PE],
                    STOP=self.config.stop_idx[const.PE],
                )
            )
            metrics.append(
                ExpectedErrorMetric(
                    prefix=const.PE,
                    target_name=const.PE,
                    PAD=self.config.pad_idx[const.PE],
                    STOP=self.config.stop_idx[const.PE],
                )
            )
            metrics.append(
                PerplexityMetric(
                    prefix=const.PE,
                    target_name=const.PE,
                    PAD=self.config.pad_idx[const.PE],
                    STOP=self.config.stop_idx[const.PE],
                )
            )
        if self.config.token_level and const.SOURCE in self.predictors:
            metrics.append(
                CorrectMetric(
                    prefix=const.SOURCE,
                    target_name=const.SOURCE,
                    PAD=self.config.pad_idx[const.SOURCE],
                    STOP=self.config.stop_idx[const.SOURCE],
                )
            )
            metrics.append(
                ExpectedErrorMetric(
                    prefix=const.SOURCE,
                    target_name=const.SOURCE,
                    PAD=self.config.pad_idx[const.SOURCE],
                    STOP=self.config.stop_idx[const.SOURCE],
                )
            )
            metrics.append(
                PerplexityMetric(
                    prefix=const.SOURCE,
                    target_name=const.SOURCE,
                    PAD=self.config.pad_idx[const.SOURCE],
                    STOP=self.config.stop_idx[const.SOURCE],
                )
            )

        return metrics

    def metrics_ordering(self):
        return max
