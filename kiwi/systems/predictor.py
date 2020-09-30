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
from kiwi.data.encoders.parallel_data_encoder import ParallelDataEncoder
from kiwi.systems.encoders.predictor import PredictorEncoder
from kiwi.systems.outputs.translation_language_model import TLMOutputs
from kiwi.systems.tlm_system import TLMSystem
from kiwi.utils.io import BaseConfig

logger = logging.getLogger(__name__)


class ModelConfig(BaseConfig):
    encoder: PredictorEncoder.Config = PredictorEncoder.Config()
    tlm_outputs: TLMOutputs.Config = TLMOutputs.Config()


@TLMSystem.register_subclass
class Predictor(TLMSystem):
    """Predictor TLM, used for the Predictor-Estimator QE model (proposed in 2017)."""

    class Config(TLMSystem.Config):
        model: ModelConfig

    def __init__(
        self,
        config: Config,
        data_config: WMTQEDataset.Config = None,
        module_dict: Dict[str, Any] = None,
    ):
        super().__init__(config, data_config=data_config)

        if module_dict:
            # Load modules and weights
            self._load_dict(module_dict)
        else:
            # Initialize data processing
            self.data_encoders = ParallelDataEncoder(
                config=self.config.data_processing,
                field_encoders=PredictorEncoder.input_data_encoders(
                    self.config.model.encoder
                ),
            )

        # Add possibly missing fields, like outputs
        if self.config.load_vocabs:
            self.data_encoders.load_vocabularies(self.config.load_vocabs)
        if self.train_dataset:
            self.data_encoders.fit_vocabularies(self.train_dataset)

        # Input to features
        if not self.encoder:
            self.encoder = PredictorEncoder(
                vocabs=self.data_encoders.vocabularies,
                config=self.config.model.encoder,
                pretraining=True,
            )

        # Output
        if not self.tlm_outputs:
            self.tlm_outputs = TLMOutputs(
                inputs_dims=self.encoder.size(),
                vocabs=self.data_encoders.vocabularies,
                config=self.config.model.tlm_outputs,
                pretraining=True,
            )
