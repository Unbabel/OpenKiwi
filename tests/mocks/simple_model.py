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
import torch

from kiwi import constants as const
from kiwi.metrics import LogMetric
from kiwi.models.model import Model
from kiwi.systems.decoders.quetch_deprecated import QUETCH
from kiwi.systems.encoders.quetch import QUETCHEncoder


@Model.register_subclass
class SimpleModel(Model):

    target = const.TARGET_TAGS
    fieldset = QUETCH.fieldset
    metrics_ordering = QUETCH.metrics_ordering

    @staticmethod
    def default_features_embedder_class():
        return QUETCHEncoder

    def _build_output_layer(self, vocabs, features_dim, config):
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(features_dim, 1), torch.nn.Sigmoid()
        )

    def forward(self, batch, *args, **kwargs):
        field_embeddings = {
            field: self.field_embedders[field](batch[field])
            for field in (const.SOURCE, const.TARGET)
        }
        feature_embeddings = self.encoder(field_embeddings, batch)
        output = self.output_layer(feature_embeddings).squeeze()
        return {SimpleModel.target: output}

    def loss(self, model_out, batch):
        prediction = model_out[SimpleModel.target]
        target = getattr(batch, SimpleModel.target).float()
        return {const.LOSS: ((prediction - target) ** 2).mean()}

    def predict(self, batch, *args, **kwargs):
        predictions = self.forward(batch)
        return {SimpleModel.target: predictions[SimpleModel.target].tolist()}

    def metrics(self):
        return (LogMetric(log_targets=[(const.LOSS, const.LOSS)]),)
