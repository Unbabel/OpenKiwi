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

import os.path
import warnings
from pathlib import Path

import numpy as np
from more_itertools import flatten
from scipy.stats.stats import pearsonr, rankdata, spearmanr

from kiwi import constants as const
from kiwi.data.utils import read_file
from kiwi.metrics.functions import (
    delta_average,
    f1_scores,
    mean_absolute_error,
    mean_squared_error,
)


def evaluate_from_options(options):
    """
    Evaluates a model's predictions based on the flags received from
     the configuration files. Refer to configuration for a list of
    available configuration flags for the evaluate pipeline.

    Args:
        options (Namespace): Namespace containing all pipeline options
    """

    if options is None:
        return

    setup()
    pipeline_options = options.pipeline

    # flag denoting format so there's no need to always check
    is_wmt18_format = pipeline_options.format.lower() == "wmt18"
    is_wmt18_pred_format = pipeline_options.pred_format.lower() == "wmt18"

    # handling of gold target
    golds = retrieve_gold_standard(pipeline_options, is_wmt18_format)

    # handling of prediction files
    pred_files = retrieve_predictions(pipeline_options, is_wmt18_pred_format)

    if not any(pred_files.values()):
        print(
            "Please specify at least one of these options: "
            "--input-dir, --pred-target, --pred-source, --pred-sents"
        )
        return

    # evaluate word level
    for tag in const.TAGS:
        if tag in golds and pred_files[tag]:
            scores = eval_word_level(golds, pred_files, tag)
            print_scores_table(scores, tag)

    # evaluate sentence level
    if const.SENTENCE_SCORES in golds:
        sent_golds = golds[const.SENTENCE_SCORES]
        sent_preds = retrieve_sentence_predictions(pipeline_options, pred_files)

        if sent_preds:
            sentence_scores, sentence_ranking = eval_sentence_level(
                sent_golds, sent_preds
            )
            print_sentences_scoring_table(sentence_scores)
            print_sentences_ranking_table(sentence_ranking)

        for pred_file in pred_files[const.BINARY]:
            sent_preds.append(pred_file)

    teardown()

    # TODO return some evaluation info besides just printing the graph


def retrieve_gold_standard(pipeline_options, is_wmt18_format):
    golds = {}
    if pipeline_options.gold_target:
        gold_target = _wmt_to_labels(read_file(pipeline_options.gold_target))
        if is_wmt18_format:
            gold_target, gold_gaps = _split_wmt18(gold_target)
            golds[const.GAP_TAGS] = gold_gaps
        golds[const.TARGET_TAGS] = gold_target
    # handling of gold source
    if pipeline_options.gold_source:
        gold_source = _wmt_to_labels(read_file(pipeline_options.gold_source))
        golds[const.SOURCE_TAGS] = gold_source
    # handling of gold sentences
    if pipeline_options.gold_sents:
        gold_sentences = _read_sentence_scores(pipeline_options.gold_sents)
        golds[const.SENTENCE_SCORES] = gold_sentences
    return golds


def retrieve_predictions(pipeline_options, is_wmt18_pred_format):
    pred_files = {target: [] for target in const.TARGETS}
    if pipeline_options.pred_target:
        for pred_file in pipeline_options.pred_target:
            pred_target = read_file(pred_file)
            if is_wmt18_pred_format:
                pred_target, pred_gaps = _split_wmt18(pred_target)
                pred_files[const.GAP_TAGS].append((str(pred_file), pred_gaps))
            pred_files[const.TARGET_TAGS].append((str(pred_file), pred_target))
    if pipeline_options.pred_gaps:
        for pred_file in pipeline_options.pred_gaps:
            pred_gaps = read_file(pred_file)
            pred_files[const.GAP_TAGS].append((str(pred_file), pred_gaps))
    if pipeline_options.pred_source:
        for pred_file in pipeline_options.pred_source:
            pred_source = read_file(pred_file)
            pred_files[const.SOURCE_TAGS].append((str(pred_file), pred_source))
    if pipeline_options.pred_sents:
        for pred_file in pipeline_options.pred_sents:
            pred_sents = _read_sentence_scores(pred_file)
            pred_files[const.SENTENCE_SCORES].append(
                (str(pred_file), pred_sents)
            )
    if pipeline_options.input_dir:
        for input_dir in pipeline_options.input_dir:
            input_dir = Path(input_dir)
            for target in const.TAGS:
                pred_file = input_dir.joinpath(target)
                if pred_file.exists() and pred_file.is_file():
                    pred_files[pred_file.name].append(
                        (str(pred_file), read_file(pred_file))
                    )
            for target in [const.SENTENCE_SCORES, const.BINARY]:
                pred_file = input_dir.joinpath(target)
                if pred_file.exists() and pred_file.is_file():
                    pred_files[pred_file.name].append(
                        (str(pred_file), _read_sentence_scores(str(pred_file)))
                    )

    # Numericalize Text Labels
    if pipeline_options.type == "tags":
        for tag_name in const.TAGS:
            for i in range(len(pred_files[tag_name])):
                fname, pred_tags = pred_files[tag_name][i]
                pred_files[tag_name][i] = (fname, _wmt_to_labels(pred_tags))
    return pred_files


def retrieve_sentence_predictions(pipeline_options, pred_files):
    sent_preds = pred_files[const.SENTENCE_SCORES]
    sents_avg = (
        pipeline_options.sents_avg
        if pipeline_options.sents_avg
        else pipeline_options.type
    )
    tag_to_sent = _probs_to_sentence_score
    if sents_avg == "tags":
        tag_to_sent = _tags_to_sentence_score
    for pred_file, pred in pred_files[const.TARGET_TAGS]:
        sent_pred = np.array(tag_to_sent(pred))
        sent_preds.append((pred_file, sent_pred))
    return sent_preds


def _split_wmt18(tags):
    """Split tags list of lists in WMT18 format into target and gap tags."""
    tags_mt = [sent_tags[1::2] for sent_tags in tags]
    tags_gaps = [sent_tags[::2] for sent_tags in tags]
    return tags_mt, tags_gaps


def _wmt_to_labels(corpus):
    """Generates numeric labels from text labels."""
    dictionary = dict(zip(const.LABELS, range(len(const.LABELS))))
    return [[dictionary[word] for word in sent] for sent in corpus]


def _read_sentence_scores(sent_file):
    """Read File with numeric scores for sentences."""
    return np.array([line.strip() for line in open(sent_file)], dtype="float")


def _tags_to_sentence_score(tags_sentences):
    scores = []
    bad_label = const.LABELS.index(const.BAD)
    for tags in tags_sentences:
        labels = _probs_to_labels(tags)
        scores.append(labels.count(bad_label) / len(tags))
    return scores


def _probs_to_sentence_score(probs_sentences):
    scores = []
    for probs in probs_sentences:
        probs = [float(p) for p in probs]
        scores.append(np.mean(probs))
    return scores


def _probs_to_labels(probs, threshold=0.5):
    """Generates numeric labels from probabilities.

    This assumes two classes and default decision threshold 0.5

    """
    return [int(float(prob) > threshold) for prob in probs]


def _check_lengths(gold, prediction):
    for i, (g, p) in enumerate(zip(gold, prediction)):
        if len(g) != len(p):
            warnings.warn(
                "Mismatch length for {}th sample "
                "{} x {}".format(i, len(g), len(p))
            )


def _average(probs_per_file):
    # flat_probs = [list(flatten(probs)) for probs in probs_per_file]
    probabilities = np.array(probs_per_file, dtype="float32")
    return probabilities.mean(axis=0).tolist()


def _extract_path_prefix(file_names):
    if len(file_names) < 2:
        return "", file_names
    prefix_path = os.path.commonpath(
        [path for path in file_names if not path.startswith("*")]
    )
    if len(prefix_path) > 0:
        file_names = [
            os.path.relpath(path, prefix_path)
            if not path.startswith("*")
            else path
            for path in file_names
        ]
    return prefix_path, file_names


def setup():
    pass


def teardown():
    pass


def eval_word_level(golds, pred_files, tag_name):
    scores_table = []
    for pred_file, pred in pred_files[tag_name]:
        _check_lengths(golds[tag_name], pred)

        scores = score_word_level(
            list(flatten(golds[tag_name])), list(flatten(pred))
        )

        scores_table.append((pred_file, *scores))
    # If more than one system is provided, compute ensemble score
    if len(pred_files[tag_name]) > 1:
        ensemble_pred = _average(
            [list(flatten(pred)) for _, pred in pred_files[tag_name]]
        )
        ensemble_score = score_word_level(
            list(flatten(golds[tag_name])), ensemble_pred
        )
        scores_table.append(("*ensemble*", *ensemble_score))

    scores = np.array(
        scores_table,
        dtype=[
            ("File", "object"),
            ("F1_{}".format(const.LABELS[0]), float),
            ("F1_{}".format(const.LABELS[1]), float),
            ("F1_mult", float),
        ],
    )
    # Put the main metric in the first column
    scores = scores[
        [
            "File",
            "F1_mult",
            "F1_{}".format(const.LABELS[0]),
            "F1_{}".format(const.LABELS[1]),
        ]
    ]

    return scores


def eval_sentence_level(sent_gold, sent_preds):
    sentence_scores, sentence_ranking = [], []
    for file_name, pred in sent_preds:
        scoring, ranking = score_sentence_level(sent_gold, pred)

        sentence_scores.append((file_name, *scoring))
        sentence_ranking.append((file_name, *ranking))

    ensemble_pred = _average([pred for _, pred in sent_preds])
    ensemble_score, ensemble_ranking = score_sentence_level(
        sent_gold, ensemble_pred
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


def score_word_level(gold, prediction):
    gold_tags = gold
    pred_tags = _probs_to_labels(prediction)
    return f1_scores(pred_tags, gold_tags)


def score_sentence_level(gold, pred):
    pearson = pearsonr(gold, pred)
    mae = mean_absolute_error(gold, pred)
    rmse = np.sqrt(mean_squared_error(gold, pred))

    spearman = spearmanr(
        rankdata(gold, method="ordinal"), rankdata(pred, method="ordinal")
    )
    delta_avg = delta_average(gold, rankdata(pred, method="ordinal"))

    return (pearson[0], mae, rmse), (spearman[0], delta_avg)


def print_scores_table(scores, prefix="TARGET"):
    prefix_path, scores["File"] = _extract_path_prefix(scores["File"])
    path_str = " ({})".format(prefix_path) if prefix_path else ""

    max_method_length = max(len(path_str) + 4, max(map(len, scores["File"])))
    print("-" * (max_method_length + 13 * 3))
    print("Word-level scores for {}:".format(prefix))
    print(
        "{:{width}}    {:9}    {:9}    {:9}".format(
            "File{}".format(path_str),
            "F1_mult",
            "F1_{}".format(const.LABELS[0]),
            "F1_{}".format(const.LABELS[1]),
            width=max_method_length,
        )
    )
    for score in np.sort(scores, order=["F1_mult", "File"])[::-1]:
        print(
            "{:{width}s}    {:<9.5f}    {:<9.5}    {:<9.5f}".format(
                *score, width=max_method_length
            )
        )


def print_sentences_scoring_table(scores):
    prefix_path, scores["File"] = _extract_path_prefix(scores["File"])
    path_str = " ({})".format(prefix_path) if prefix_path else ""

    max_method_length = max(len(path_str) + 4, max(map(len, scores["File"])))
    print("-" * (max_method_length + 13 * 3))
    print("Sentence-level scoring:")
    print(
        "{:{width}}    {:9}    {:9}    {:9}".format(
            "File{}".format(path_str),
            "Pearson r",
            "MAE",
            "RMSE",
            width=max_method_length,
        )
    )
    for score in np.sort(scores, order=["Pearson r", "File"])[::-1]:
        print(
            "{:{width}s}    {:<9.5f}    {:<9.5f}    {:<9.5f}".format(
                *score, width=max_method_length
            )
        )


def print_sentences_ranking_table(scores):
    prefix_path, scores["File"] = _extract_path_prefix(scores["File"])
    path_str = " ({})".format(prefix_path) if prefix_path else ""

    max_method_length = max(len(path_str) + 4, max(map(len, scores["File"])))
    print("-" * (max_method_length + 13 * 3))
    print("Sentence-level ranking:")
    print(
        "{:{width}}    {:10}    {:9}".format(
            "File{}".format(path_str),
            "Spearman r",
            "DeltaAvg",
            width=max_method_length,
        )
    )  # noqa
    for score in np.sort(scores, order=["Spearman r", "File"])[::-1]:
        print(
            "{:{width}s}    {:<10.5f}    {:<9.5f}".format(
                *score, width=max_method_length
            )
        )
