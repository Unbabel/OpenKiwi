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
from typing import Any, Dict

from kiwi.data.datasets.wmt_qe_dataset import WMTQEDataset
from kiwi.data.encoders.wmt_qe_data_encoder import WMTQEDataEncoder
from kiwi.systems.decoders.linear import LinearDecoder
from kiwi.systems.encoders.xlmroberta import XLMRobertaEncoder
from kiwi.systems.outputs.quality_estimation import QEOutputs
from kiwi.systems.outputs.translation_language_model import TLMOutputs
from kiwi.systems.qe_system import QESystem
from kiwi.utils.io import BaseConfig

logger = logging.getLogger(__name__)


class ModelConfig(BaseConfig):
    encoder: XLMRobertaEncoder.Config = XLMRobertaEncoder.Config()
    decoder: LinearDecoder.Config = LinearDecoder.Config()
    outputs: QEOutputs.Config = QEOutputs.Config()
    tlm_outputs: TLMOutputs.Config = TLMOutputs.Config()


@QESystem.register_subclass
class XLMRoberta(QESystem):
    """XLMRoberta-based Predictor-Estimator model"""

    class Config(QESystem.Config):
        model: ModelConfig

    def __init__(
        self,
        config,
        data_config: WMTQEDataset.Config = None,
        module_dict: Dict[str, Any] = None,
    ):
        super().__init__(config, data_config=data_config)

        if module_dict:
            # Load modules and weights
            self._load_dict(module_dict)
        elif self.config.load_encoder:
            self._load_encoder(self.config.load_encoder)
        else:
            # Initialize data processing
            self.data_encoders = WMTQEDataEncoder(
                config=self.config.data_processing,
                field_encoders=XLMRobertaEncoder.input_data_encoders(
                    self.config.model.encoder
                ),
            )

        # Add possibly missing fields, like outputs
        if self.config.load_vocabs:
            self.data_encoders.load_vocabularies(self.config.load_vocabs)
        if self.train_dataset:
            self.data_encoders.fit_vocabularies(self.train_dataset)
        # else:
        #     raise ValueError(
        #         'No training data available and no pretrained vocabulary loaded; '
        #         'pass a data configuration or `load_encoder` or `load_vocabs`'
        #     )

        # Input to features
        if not self.encoder:
            self.encoder = XLMRobertaEncoder(
                vocabs=self.data_encoders.vocabularies, config=self.config.model.encoder
            )

        # Features to output
        if not self.decoder:
            self.decoder = LinearDecoder(
                inputs_dims=self.encoder.size(), config=self.config.model.decoder
            )

        # Output layers
        if not self.outputs:
            self.outputs = QEOutputs(
                inputs_dims=self.decoder.size(),
                vocabs=self.data_encoders.vocabularies,
                config=self.config.model.outputs,
            )

        if not self.tlm_outputs:
            self.tlm_outputs = TLMOutputs(
                inputs_dims=self.encoder.size(),
                vocabs=self.data_encoders.vocabularies,
                config=self.config.model.tlm_outputs,
            )
