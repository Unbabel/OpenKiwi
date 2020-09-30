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
import os.path
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from more_itertools import collapse
from pydantic import FilePath
from pydantic.class_validators import validator
from scipy.stats.stats import pearsonr, rankdata, spearmanr

from kiwi import constants as const
from kiwi.data.datasets import wmt_qe_dataset
from kiwi.metrics.functions import (
    delta_average,
    matthews_correlation_coefficient,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from kiwi.utils.io import (
    BaseConfig,
    read_file,
    target_gaps_to_gaps,
    target_gaps_to_target,
)

logger = logging.getLogger(__name__)


class OutputConfig(wmt_qe_dataset.OutputConfig):
    gap_tags: Optional[FilePath] = None
    '''Path to label file for gaps (only for predictions)'''
    targetgaps_tags: Optional[FilePath] = None
    '''Path to label file for target+gaps (only for predictions)'''


class Configuration(BaseConfig):
    gold_files: wmt_qe_dataset.OutputConfig
    predicted_files: Optional[List[OutputConfig]]
    predicted_dir: Optional[List[Path]] = None
    '''One or more directories from where to read predicted files (using standard output
    names.'''

    @validator(
        'predicted_files', 'predicted_dir', pre=True, always=False, each_item=False
    )
    def ensure_list(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            v = [v]
        return v

    @validator('predicted_dir', pre=False, always=True)
    def check_consistency(cls, v, values):
        if not v and not values['predicted_files']:
            raise ValueError('Must provide either `predicted_files` or `predicted_dir`')
        return v


@dataclass
class MetricsReport:
    word_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    sentence_scores: Dict[str, np.ndarray] = field(default_factory=dict)

    def add_word_level_scores(self, name: str, scores: np.ndarray):
        self.word_scores[name] = scores

    def add_sentence_level_scores(self, name: str, scores: np.ndarray):
        self.sentence_scores[name] = scores

    def print_scores_table(self):
        print(self)

    def __str__(self):
        string = ''
        for name, scores in self.word_scores.items():
            scores_string = self._scores_str(scores)
            width = len(scores_string.split('\n', maxsplit=1)[0])
            string += '-' * width + '\n'
            string += f'Word-level scores for {name}:\n'
            string += scores_string
        for name, scores in self.sentence_scores.items():
            scores_string = self._scores_str(scores)
            width = len(scores_string.split('\n', maxsplit=1)[0])
            string += '-' * width + '\n'
            string += f'Sentence-level {name}:\n'
            string += scores_string
        return string

    @staticmethod
    def _scores_str(scores: np.ndarray) -> str:
        column_names = scores.dtype.names
        first_col = column_names[0]
        prefix_path, scores[first_col] = _extract_path_prefix(scores[first_col])
        if prefix_path:
            path_str = f' ({prefix_path})'
        else:
            path_str = ''

        max_method_length = max(len(path_str) + 4, max(map(len, scores[first_col])))
        template = '{:{width}}' + '    {:9}' * len(column_names[1:])
        header = template.format(
            f'{first_col}{path_str}', *scores.dtype.names[1:], width=max_method_length,
        )
        string = f'{header}\n'
        row_template = '{:{width}s}' + '    {:<9.5f}' * len(column_names[1:])
        for score in np.sort(scores, order=[column_names[1], first_col])[::-1]:
            row = row_template.format(*score, width=max_method_length)
            string = f'{string}{row}\n'

        return string


def evaluate_from_configuration(configuration_dict: Dict[str, Any]):
    """Evaluate a model's predictions based on the flags received from the configuration
    files.

    Refer to configuration for a list of available configuration flags for the evaluate
    pipeline.

    Args:
        configuration_dict: options read from file or CLI
    """
    config = Configuration(**configuration_dict)
    report = run(config)
    print(report)


def run(config: Configuration) -> MetricsReport:
    """Runs the evaluation pipeline for evaluating a model's predictions. Essentially
    calculating metrics using `gold_targets` and `prediction_files`.

    Refer to configuration for a list of available options for this pipeline.

    Args:
        config: Configuration Namespace

    return:
        MetricsReport: Object with information for both word and sentence level metrics

    """

    gold_targets = retrieve_gold_standard(config.gold_files)
    prediction_files = normalize_prediction_files(
        config.predicted_files, config.predicted_dir
    )

    all_scores = MetricsReport()

    # evaluate word level
    source_predictions = {}
    if const.SOURCE_TAGS in gold_targets and const.SOURCE_TAGS in prediction_files:
        true_targets = gold_targets[const.SOURCE_TAGS]
        for file_name in prediction_files[const.SOURCE_TAGS]:
            predicted_targets = to_numeric_values(read_file(file_name))
            if not lengths_match(true_targets, predicted_targets):
                report_lengths_mismatch(true_targets, predicted_targets)
                logger.warning(f'Skipping {file_name}')
                continue
            source_predictions[file_name] = predicted_targets
        if source_predictions:
            scores = eval_word_level(true_targets, source_predictions)
            all_scores.add_word_level_scores(const.SOURCE_TAGS, scores)

    targetgaps_predictions = {}
    target_predictions = {}
    gap_predictions = {}
    if const.TARGET_TAGS in gold_targets and (
        const.TARGET_TAGS in prediction_files
        or const.TARGETGAPS_TAGS in prediction_files
        or const.GAP_TAGS in prediction_files
    ):
        true_targets = gold_targets[const.TARGET_TAGS]
        if all([len(sentence) % 2 == 1 for sentence in true_targets]):
            # It's probably target+gaps
            true_target_targets, true_gap_targets = split_wmt18_tags(true_targets)
        else:
            true_target_targets = true_gap_targets = []

        for file_name in prediction_files.get(
            const.TARGET_TAGS, []
        ) + prediction_files.get(const.TARGETGAPS_TAGS, []):
            predicted_targets = to_numeric_values(read_file(file_name))
            if lengths_match(true_targets, predicted_targets):
                targetgaps_predictions[file_name] = predicted_targets
                if true_target_targets and true_gap_targets:
                    predicted_target, predicted_gaps = split_wmt18_tags(
                        predicted_targets
                    )
                    target_predictions[file_name] = predicted_target
                    gap_predictions[file_name] = predicted_gaps
            elif true_target_targets and lengths_match(
                true_target_targets, predicted_targets
            ):
                target_predictions[file_name] = predicted_targets
            else:
                report_lengths_mismatch(true_targets, predicted_targets)
                logger.warning(f'Skipping {file_name}')
                continue
        if true_gap_targets:
            for file_name in prediction_files.get(const.GAP_TAGS, []):
                predicted_targets = to_numeric_values(read_file(file_name))
                if lengths_match(true_gap_targets, predicted_targets):
                    gap_predictions[file_name] = predicted_targets

        if targetgaps_predictions:
            scores = eval_word_level(true_targets, targetgaps_predictions)
            all_scores.add_word_level_scores(const.TARGETGAPS_TAGS, scores)
        if target_predictions:
            scores = eval_word_level(true_target_targets, target_predictions)
            all_scores.add_word_level_scores(const.TARGET_TAGS, scores)
        if gap_predictions:
            scores = eval_word_level(true_gap_targets, gap_predictions)
            all_scores.add_word_level_scores(const.GAP_TAGS, scores)

    # evaluate sentence level
    sentence_predictions = {}
    if const.SENTENCE_SCORES in gold_targets:
        true_targets = gold_targets[const.SENTENCE_SCORES]

        if const.SENTENCE_SCORES in prediction_files:
            for file_name in prediction_files[const.SENTENCE_SCORES]:
                predicted_targets = read_sentence_scores_file(file_name)
                sentence_predictions[file_name] = predicted_targets

        if target_predictions:
            for file_name, predicted_targets in target_predictions.items():
                sentence_predictions[file_name] = [
                    np.mean(word_predictions, keepdims=False)
                    for word_predictions in predicted_targets
                ]

        if sentence_predictions:
            sentence_scores, sentence_ranking = eval_sentence_level(
                true_targets, sentence_predictions
            )
            all_scores.add_sentence_level_scores('scoring', sentence_scores)
            all_scores.add_sentence_level_scores('ranking', sentence_ranking)

    return all_scores


def retrieve_gold_standard(config: OutputConfig):
    golds = {}
    if config.target_tags:
        # gold_target = _wmt_to_labels(read_file(config.target_tags))
        gold_target = to_numeric_binary_labels(read_file(config.target_tags))
        golds[const.TARGET_TAGS] = gold_target
    # handling of gold source
    if config.source_tags:
        gold_source = to_numeric_binary_labels(read_file(config.source_tags))
        golds[const.SOURCE_TAGS] = gold_source
    # handling of gold sentences
    if config.sentence_scores:
        gold_sentences = read_sentence_scores_file(config.sentence_scores)
        golds[const.SENTENCE_SCORES] = gold_sentences
    return golds


def normalize_prediction_files(
    predicted_files_config: List[OutputConfig], predicted_dir_config: List[Path],
):
    prediction_files = defaultdict(list)
    if predicted_files_config:
        for prediction_set_config in predicted_files_config:
            if prediction_set_config.source_tags:
                prediction_files[const.SOURCE_TAGS].append(
                    prediction_set_config.source_tags
                )
            if prediction_set_config.target_tags:
                prediction_files[const.TARGET_TAGS].append(
                    prediction_set_config.target_tags
                )
            if prediction_set_config.gap_tags:
                prediction_files[const.GAP_TAGS].append(prediction_set_config.gap_tags)
            if prediction_set_config.targetgaps_tags:
                prediction_files[const.TARGETGAPS_TAGS].append(
                    prediction_set_config.targetgaps_tags
                )
            if prediction_set_config.sentence_scores:
                prediction_files[const.SENTENCE_SCORES].append(
                    prediction_set_config.sentence_scores
                )

    if predicted_dir_config:
        for input_dir in predicted_dir_config:
            for file_name in [
                const.SOURCE_TAGS,
                const.TARGET_TAGS,
                const.GAP_TAGS,
                const.TARGETGAPS_TAGS,
                const.SENTENCE_SCORES,
            ]:
                predictions_file = input_dir.joinpath(file_name)
                if predictions_file.exists() and predictions_file.is_file():
                    prediction_files[file_name].append(predictions_file)

    return prediction_files


def split_wmt18_tags(tags: List[List[Any]]):
    """Split tags list of lists in WMT18 format into target and gap tags."""
    tags_mt = [target_gaps_to_target(sent_tags) for sent_tags in tags]
    tags_gaps = [target_gaps_to_gaps(sent_tags) for sent_tags in tags]
    return tags_mt, tags_gaps


def read_sentence_scores_file(sent_file):
    """Read file with numeric scores for sentences."""
    return np.array([line.strip() for line in open(sent_file)], dtype="float")


def to_numeric_values(
    predictions: Union[str, List[str], List[List[str]]]
) -> Union[int, float, List[int], List[float], List[List[int]], List[List[float]]]:
    """Convert text labels or string probabilities (for BAD) to int or float values,
    respectively."""

    if isinstance(predictions, list):
        return [to_numeric_values(element) for element in predictions]
    else:
        try:
            return float(predictions)
        except ValueError as e:
            if predictions in const.LABELS:
                return const.LABELS.index(predictions)
            else:
                raise e


def to_numeric_binary_labels(
    predictions: Union[
        str, float, List[str], List[List[str]], List[float], List[List[float]]
    ],
    threshold: float = 0.5,
):
    """Generate numeric labels from text labels or probabilities (for BAD)."""
    if isinstance(predictions, list):
        return [to_numeric_binary_labels(element) for element in predictions]
    else:
        try:
            return int(float(predictions) > threshold)
        except ValueError as e:
            if predictions in const.LABELS:
                return const.LABELS.index(predictions)
            else:
                raise e


def report_lengths_mismatch(gold, prediction):
    """Checks if the number of gold and predictions labels match. Prints a warning and
    returns false if they do not.

    Args:
        gold: list of gold labels
        prediction: list of predicted labels

    Return:
        bool: True if all lenghts match, False if not
    """
    for i, (g, p) in enumerate(zip(gold, prediction)):
        if len(g) != len(p):
            logger.warning(f'Mismatch length for {i}th sample {len(g)} x {len(p)}')
            return False
    return True


def lengths_match(gold, prediction):
    """Checks if the number of gold and predictions labels match. Returns false if they
     do not.

    Args:
        gold: list of gold labels
        prediction: list of predicted labels

    Return:
        bool: True if all lenghts match, False if not
    """
    for i, (g, p) in enumerate(zip(gold, prediction)):
        if len(g) != len(p):
            return False
    return True


def word_level_scores(true_targets, predicted_targets, labels=const.LABELS):
    y = list(collapse(true_targets))
    y_hat = list(collapse(to_numeric_binary_labels(predicted_targets)))
    p, r, f1, s = precision_recall_fscore_support(y_hat, y, labels=labels)
    f1_mult = np.prod(f1)
    mcc = matthews_correlation_coefficient(y_hat, y)

    scores = {
        'MCC': mcc,
        'F1_Mult': f1_mult,
    }
    for i in range(len(labels)):
        scores[f'F1_{labels[i]}'] = f1[i]

    return scores


def eval_word_level(
    true_targets, predictions: Dict[str, List[List[int]]]
) -> np.ndarray:
    scores_table = None

    for file_name, predicted_targets in predictions.items():
        scores = word_level_scores(true_targets, predicted_targets)

        dtype = dict(
            names=['File'] + list(scores.keys()),
            formats=['object'] + ['f8'] * len(scores),
        )
        scores_array = np.array(
            [(str(file_name),) + tuple(scores.values())], dtype=dtype
        )
        if scores_table is not None:
            scores_table = np.concatenate((scores_table, scores_array))
        else:
            scores_table = scores_array

    # If more than one system is provided, compute ensemble score
    if len(predictions) > 1:
        ensemble_targets = np.array(
            [
                list(collapse(file_predictions))
                for file_predictions in predictions.values()
            ],
            dtype='float32',
        )
        ensemble_targets = ensemble_targets.mean(axis=0).tolist()

        scores = word_level_scores(true_targets, ensemble_targets)
        dtype = dict(
            names=['File'] + list(scores.keys()),
            formats=['object'] + ['f8'] * len(scores),
        )
        scores_array = np.array([('*ensemble*',) + tuple(scores.values())], dtype=dtype)
        scores_table = np.concatenate((scores_table, scores_array))

    return scores_table


def sentence_level_scores(
    true_targets: List[float], predicted_targets: List[float]
) -> Tuple[Tuple, Tuple]:
    pearson = pearsonr(true_targets, predicted_targets)
    mae = mean_absolute_error(true_targets, predicted_targets)
    rmse = np.sqrt(mean_squared_error(true_targets, predicted_targets))

    spearman = spearmanr(
        rankdata(true_targets, method="ordinal"),  # NOQA
        rankdata(predicted_targets, method="ordinal"),  # NOQA
    )
    delta_avg = delta_average(
        true_targets, rankdata(predicted_targets, method="ordinal")
    )

    return (pearson[0], mae, rmse), (spearman[0], delta_avg)


def eval_sentence_level(
    true_targets, predictions: Dict[str, List[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    sentence_scores, sentence_ranking = [], []
    for file_name, predicted_target in predictions.items():
        scoring, ranking = sentence_level_scores(true_targets, predicted_target)

        sentence_scores.append((str(file_name), *scoring))
        sentence_ranking.append((str(file_name), *ranking))

    if len(predictions) > 1:
        ensemble_targets = np.array(list(predictions.values()), dtype='float32')
        ensemble_targets = ensemble_targets.mean(axis=0).tolist()
        ensemble_score, ensemble_ranking = sentence_level_scores(
            true_targets, ensemble_targets
        )
        sentence_scores.append(("*ensemble*", *ensemble_score))
        sentence_ranking.append(("*ensemble*", *ensemble_ranking))

    sentence_scores = np.array(
        sentence_scores,
        dtype=[
            ("File", "object"),
            ("Pearson r", float),
            ("MAE", float),
            ("RMSE", float),
        ],
    )
    sentence_ranking = np.array(
        sentence_ranking,
        dtype=[("File", "object"), ("Spearman r", float), ("DeltaAvg", float)],
    )

    return sentence_scores, sentence_ranking


def _extract_path_prefix(file_names):
    if len(file_names) < 2:
        return "", file_names
    prefix_path = os.path.commonpath(
        [path for path in file_names if not path.startswith("*")]
    )
    if len(prefix_path) > 0:
        file_names = [
            os.path.relpath(path, prefix_path) if not path.startswith("*") else path
            for path in file_names
        ]
    return prefix_path, file_names
