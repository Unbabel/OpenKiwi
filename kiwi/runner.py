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
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from kiwi import constants as const
from kiwi.data.batch import tensors_to
from kiwi.data.datasets.wmt_qe_dataset import WMTQEDataset
from kiwi.systems.outputs.quality_estimation import QEOutputs
from kiwi.systems.qe_system import QESystem

logger = logging.getLogger(__name__)


@dataclass
class Predictions:
    sentences_hter: List[float] = None

    target_tags_BAD_probabilities: List[List[float]] = None
    target_tags_labels: List[List[str]] = None

    source_tags_BAD_probabilities: List[List[float]] = None
    source_tags_labels: List[List[str]] = None

    gap_tags_BAD_probabilities: List[List[float]] = None
    gap_tags_labels: List[List[str]] = None


class Runner:
    def __init__(self, system: QESystem, device: Optional[int] = None):
        """Class to load a system for inference.

        Arguments:
            system: a trained QE system
            device: to use when loading and running the system; pass a GPU id or
                    ``None`` (default) for using CPU.
        """
        self.system = system
        self.device = device

        self.system.freeze()

        if self.device is not None:
            torch.cuda.set_device(self.device)
        self.system.to(self.device)

    def configure_outputs(self, output_config: QEOutputs.Config):
        if output_config:
            self.system.outputs.config = output_config

    @staticmethod
    def wrap_predictions(predictions: Dict[str, List[Any]]) -> Predictions:
        mapping = dict(
            sentences_hter=const.SENTENCE_SCORES,
            target_tags_BAD_probabilities=const.TARGET_TAGS,
            target_tags_labels=f'{const.TARGET_TAGS}_labels',
            source_tags_BAD_probabilities=const.SOURCE_TAGS,
            source_tags_labels=f'{const.SOURCE_TAGS}_labels',
            gap_tags_BAD_probabilities=const.GAP_TAGS,
            gap_tags_labels=f'{const.GAP_TAGS}_labels',
        )
        predictions_renamed = {
            new_key: predictions.get(old_key) for new_key, old_key in mapping.items()
        }
        return Predictions(**predictions_renamed)

    def predict(
        self,
        source: List[str],
        target: List[str],
        alignments: List[str] = None,
        batch_size: int = 1,
        num_data_workers: int = 0,
    ) -> Predictions:
        """Create predictions for a list of examples.

        Arguments:
            source: a list of sentences on a source language.
            target: a list of (translated) sentences on a target language.
            alignments: optional list of source-target alignments required only by the
                        NuQE model.
            batch_size: how large to build a batch (default: 1).
            num_data_workers: how many subprocesses to use for data loading.

        Return:
            A ``Predictions`` object with predicted outputs for each example in the
            inputs. If input ``source`` and ``target`` are all empty, returned object
            has all attributes as ``None``. If there are aligned empty sentences at
            both ``source`` and ``target``, the corresponding returned prediction will
            contain empty/zero values (empty list for word level outputs, 0.0 for
            sentence level outputs).

        Raise:
             Exception: If an example has an empty string as `source` xor `target`
            field (not both at the same time).

        Notes:
            ``source`` and ``target`` lenghts must match.

        Example:
            >>> from kiwi.lib import predict
            >>> runner = predict.load_system('../tests/toy-data/models/nuqe.ckpt')
            >>> source = ['a b c', 'd e f g']
            >>> target = ['q w e r', 't y']
            >>> alignments = ['0-0 1-1 1-2', '1-1 3-0']
            >>> predictions = runner.predict(source, target, alignments)
            >>> predictions.target_tags_BAD_probabilities  # doctest: +ELLIPSIS
            [[0.49699464440345764, 0.49956727027893066, ...], [..., 0.5013138651847839]]

            Predictions(
                   sentences_hter=[0.2668147683143616, 0.26675286889076233],
                   target_tags_BAD_probabilities=[
                       [
                           0.49699464440345764,
                           0.49956727027893066,
                           0.5025501847267151,
                           0.5057167410850525,
                       ],
                       [0.4967852830886841, 0.5013138651847839],
                   ],
                   target_tags_labels=[['OK', 'OK', 'BAD', 'BAD'], ['OK', 'BAD']],
                   source_tags_BAD_probabilities=None,
                   source_tags_labels=None,
                   gap_tags_BAD_probabilities=[
                       [
                           0.42644527554512024,
                           0.42096763849258423,
                           0.41709718108177185,
                           0.4157106280326843,
                           0.41496342420578003,
                       ],
                       [0.42876192927360535, 0.4251120686531067, 0.4210476577281952],
                   ],
                   gap_tags_labels=[['OK', 'OK', 'OK', 'OK', 'OK'], ['OK', 'OK', 'OK']],
            )
        """
        empty_source = empty_target = False
        if not source or all([not s.strip() for s in source]):
            empty_source = True
        if not target or all([not s.strip() for s in target]):
            empty_target = True
        if empty_source and empty_target:
            logger.warning(
                'Received empty `source` and `target` inputs. '
                'Returning empty predictions.'
            )
            return Predictions()
        elif empty_target:
            raise ValueError('Received empty `target` input, but `source` not empty.')
        elif empty_source:
            raise ValueError('Received empty `source` input, but `target` not empty.')
        if len(source) != len(target):
            raise ValueError(
                f'Number of source and target sentences must match: '
                f'{len(source)} != {len(target)}'
            )
        if alignments:
            if len(source) != len(alignments):
                logger.error(
                    f'Number of source and alignment samples must match: '
                    f'{len(source)} != {len(alignments)}'
                )

        columns = {
            const.SOURCE: source,
            const.TARGET: target,
        }
        if alignments:
            columns[const.ALIGNMENTS] = alignments

        columns, indices_of_empties = self.remove_empty_sentences(columns)

        dataset = WMTQEDataset(columns)
        dataloader = self.system.prepare_dataloader(
            dataset, batch_size=batch_size, num_workers=num_data_workers
        )

        predictions = self.run(dataloader)

        predictions = self.insert_dummy_outputs_for_empty_sentences(
            predictions, indices_of_empties
        )

        predictions = self.wrap_predictions(predictions)
        return predictions

    def run(self, iterator=None) -> Dict[str, List]:
        if iterator is None:
            iterator = self.system.test_dataloader()
        self.system.eval()
        predictions = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(
                iterator, total=len(iterator), desc='Batches', unit=' batches', ncols=80
            ):
                if self.device is not None:
                    batch = tensors_to(batch, self.device)
                sample_predictions = self.system.predict(batch)
                for key, values in sample_predictions.items():
                    if isinstance(values, list):
                        predictions[key] += values
                    else:
                        predictions[key].append(values)
        return dict(predictions)

    @staticmethod
    def remove_empty_sentences(
        columns: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[str]], List[int]]:
        new_columns = {field: [] for field in columns}
        empty_sentences_indices = []
        for i, sentences in enumerate(zip(*columns.values())):
            if any([not sentence.strip() for sentence in sentences]):
                empty_sentences_indices.append(i)
            else:
                for field, sentence in zip(columns.keys(), sentences):
                    new_columns[field].append(sentence)

        return new_columns, empty_sentences_indices

    @staticmethod
    def insert_dummy_outputs_for_empty_sentences(
        predictions: Dict[str, List], indices_of_empties: List[int]
    ) -> Dict[str, List]:
        if not indices_of_empties:
            return predictions

        first_predicted_idx = 0
        while first_predicted_idx in indices_of_empties:
            first_predicted_idx += 1

        empty_value_by_dtype = {}
        for field, field_predictions in predictions.items():
            if first_predicted_idx < len(field_predictions):
                dtype = type(field_predictions[first_predicted_idx])
            else:
                dtype = list
            if dtype == list:
                empty_value_by_dtype[field] = []
            elif dtype == float:
                empty_value_by_dtype[field] = 0.0
            elif dtype == int:
                empty_value_by_dtype[field] = 0
            elif dtype == str:
                empty_value_by_dtype[field] = ''

        new_predictions = dict(predictions)
        for missing_idx in indices_of_empties:
            for field in new_predictions:
                new_predictions[field].insert(missing_idx, empty_value_by_dtype[field])

        return new_predictions
