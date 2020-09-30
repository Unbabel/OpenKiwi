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

import torch
from pydantic import PositiveInt, confloat, validator
from torch import nn

from kiwi import constants as const
from kiwi.systems._meta_module import MetaModule
from kiwi.utils.io import BaseConfig
from kiwi.utils.tensors import apply_packed_sequence, pad_zeros_around_timesteps

logger = logging.getLogger(__name__)


@MetaModule.register_subclass
class EstimatorDecoder(MetaModule):
    class Config(BaseConfig):
        hidden_size: int = 100
        'Size of hidden layers in LSTM'

        rnn_layers: PositiveInt = 1
        'Layers in PredictorEstimator RNN'

        use_mlp: bool = True
        """Pass the PredictorEstimator input through a linear layer reducing
        dimensionality before RNN."""

        dropout: confloat(ge=0.0, le=1.0) = 0.0

        use_v0_buggy_strategy: bool = False
        """The Predictor implementation in Kiwi<=0.3.4 had a bug in applying the LSTM
        to encode source (it used lengths too short by 2) and in reversing the target
        embeddings for applying the backward LSTM (also short by 2). This flag is set
        to true when loading a saved model from those versions."""

        @validator('dropout', pre=True)
        def dropout_on_rnns(cls, v, values):
            if v > 0.0 and values['rnn_layers'] == 1:
                logger.info(
                    'Dropout on an RNN of one layer has no effect; setting it to zero.'
                )
                return 0.0
            return v

    def __init__(self, inputs_dims, config):
        super().__init__(config=config)

        self.features_dims = inputs_dims

        lstm_input_size = self.features_dims[const.TARGET]

        self.mlp = None

        # Build Model #
        if self.config.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(lstm_input_size, self.config.hidden_size), nn.Tanh()
            )
            lstm_input_size = self.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.rnn_layers,
            batch_first=True,
            dropout=self.config.dropout,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(self.config.dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        self._sizes = {
            const.TARGET: 2 * self.config.hidden_size,
            const.SOURCE: 2 * self.config.hidden_size,
        }

        # size for sentence depends if the encoder has sentence output or not
        sentence_size = 2 * self.config.rnn_layers * self.config.hidden_size
        if const.TARGET_SENTENCE in inputs_dims:
            sentence_size += inputs_dims[const.TARGET_SENTENCE]

        self._sizes[const.TARGET_SENTENCE] = sentence_size

    def size(self, field=None):
        if field:
            return self._sizes[field]
        return self._sizes

    def forward(self, features, batch_inputs):
        output_features = OrderedDict()

        if const.TARGET in features:
            features_tensor = features[const.TARGET]
            if self.mlp:
                features_tensor = self.mlp(features_tensor)

            # NOTE: this is a hackish way of finding out whether features are over
            #   pieces or over tokens.
            lengths = batch_inputs[const.TARGET].lengths
            if torch.any(lengths > features_tensor.size(1)):
                lengths = batch_inputs[const.TARGET].bounds_lengths

            # Because we need its length for the LSTM
            features_tensor, (last_hidden, last_cell) = apply_packed_sequence(
                self.lstm, features_tensor, lengths
            )
            features_tensor = self.dropout(features_tensor)

            if self.config.use_v0_buggy_strategy:
                # Add fake BOS and EOS because the output layer will remove them
                features_tensor = pad_zeros_around_timesteps(features_tensor)

            output_features[const.TARGET] = features_tensor

            """Reshape last hidden state. """
            last_hidden = last_hidden.contiguous().transpose(0, 1)
            # (layers*dir, b, dim) -> (b, layers*dir, dim)
            sentence_features = last_hidden.reshape(last_hidden.shape[0], -1)

            if const.TARGET_SENTENCE in features:
                # concat pooled_output from pre-trained models
                sentence_features = torch.cat(
                    (sentence_features, features[const.TARGET_SENTENCE]), dim=-1
                )
            # (b, layers*dir, dim) -> (b, layers*dir*dim)
            output_features[const.TARGET_SENTENCE] = sentence_features

        return output_features
