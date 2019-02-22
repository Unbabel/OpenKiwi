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
from collections import defaultdict

import torch
from torchtext.data import Example

from kiwi import constants as const
from kiwi.data.iterators import build_bucket_iterator
from kiwi.data.qe_dataset import QEDataset

logger = logging.getLogger(__name__)


class Predicter:
    def __init__(self, model, fields=None):
        """Class to load a model for inference.

        Args:
          model (kiwi.models.Model): A trained QE model
          fields (dict[str: Field]): A dict mapping field names to strings.
            For online prediction.
        """

        self.model = model
        self.fields = fields
        # Will break in Multi GPU mode
        self._device = next(model.parameters()).device

    def predict(self, examples, batch_size=1):
        """Create Predictions for a list of examples.

           Args:
             examples: A dict  mapping field names to the
               list of raw examples (strings).
             batch_size: Batch Size to use. Default 1.

           Returns:
             A dict mapping prediction levels
             (word, sentence ..) to the model predictions
             for each example.

           Raises:
             Exception: If an example has an empty string
               as `source` or `target` field.

           Example:
             >>> import kiwi
             >>> predictor = kiwi.load_model('tests/toy-data/models/nuqe.torch')
             >>> src = ['a b c', 'd e f g']
             >>> tgt = ['q w e r', 't y']
             >>> align = ['0-0 1-1 1-2', '1-1 3-0']
             >>> examples = [kiwi.constants.SOURCE: src,
                             kiwi.constants.TARGET: tgt,
                             kiwi.constants.ALIGNMENTS: align]
             >>> predictor.predict(examples)
             {'tags': [[0.4760947525501251,
                0.47569847106933594,
                0.4948718547821045,
                0.5305878520011902],
               [0.5105430483818054, 0.5252899527549744]]}
        """
        if not examples:
            return defaultdict(list)
        if self.fields is None:
            raise Exception('Missing fields object.')

        if not examples.get(const.SOURCE):
            raise KeyError('Missing required field "{}"'.format(const.SOURCE))
        if not examples.get(const.TARGET):
            raise KeyError('Missing required field "{}"'.format(const.TARGET))

        if not all(
            [s.strip() for s in examples[const.SOURCE] + examples[const.TARGET]]
        ):
            raise Exception(
                'Empty String in {} or {} field found!'.format(
                    const.SOURCE, const.TARGET
                )
            )
        fields = [(name, self.fields[name]) for name in examples]

        field_examples = [
            Example.fromlist(values, fields)
            for values in zip(*examples.values())
        ]

        dataset = QEDataset(field_examples, fields=fields)

        return self.run(dataset, batch_size)

    def run(self, dataset, batch_size=1):
        iterator = build_bucket_iterator(
            dataset, self._device, batch_size, is_train=False
        )
        self.model.eval()
        predictions = defaultdict(list)
        with torch.no_grad():
            for batch in iterator:
                model_pred = self.model.predict(batch)
                for key, values in model_pred.items():
                    if isinstance(values, list):
                        predictions[key] += values
                    else:
                        predictions[key].append(values)
        return dict(predictions)
