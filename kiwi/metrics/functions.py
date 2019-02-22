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

import numpy as np
from more_itertools import collapse

# def calibrate_threshold(scores, labels, MetricClass=LazyF1):
#     """Finds optimal decision threshold according to metric.
#     Args:
#         scores (list[float]): List of model output scores
#         labels (list): List of corresponding target labels
#     Returns:
#     (metric, threshold): The value of the Metric and the Threshold to be used.
#     """
#     metric = MetricClass(scores, labels)
#     scores, labels = metric.sort(scores, labels)
#     init_threshold = scores[0]
#     thresholds = [(metric.compute(), init_threshold)]
#     for score, label in zip(scores, labels):
#         metric.update(score, label)
#         thresholds.append((metric.compute(), score))
#     return metric.choose(thresholds)


def mean_absolute_error(y, y_hat):
    return np.mean(np.absolute(y_hat - y))


def mean_squared_error(y, y_hat):
    return np.square(np.subtract(y, y_hat)).mean()


def delta_average(y_true, y_rank):
    """Calculate the DeltaAvg score

    This is a much faster version than the Perl one provided in the
    WMT QE task 1.

    References: could not find any.

    Author: Fabio Kepler (contributed to MARMOT)

    Args:
        y_true: array of reference score (not rank) of each segment.
        y_rank: array of rank of each segment.

    Returns: the absolute delta average score.

    """
    sorted_ranked_indexes = np.argsort(y_rank)
    y_length = len(sorted_ranked_indexes)

    delta_avg = 0
    max_quantiles = y_length // 2
    set_value = (
        np.sum(y_true[sorted_ranked_indexes[np.arange(y_length)]]) / y_length
    )
    quantile_values = {
        head: np.sum(y_true[sorted_ranked_indexes[np.arange(head)]]) / head
        for head in range(2, y_length)
    }
    # Cache values, since there are many that are repeatedly computed
    # between various quantiles.
    for quantiles in range(2, max_quantiles + 1):  # Current number of quantiles
        quantile_length = y_length // quantiles
        quantile_sum = 0
        for head in np.arange(
            quantile_length, quantiles * quantile_length, quantile_length
        ):
            quantile_sum += quantile_values[head]
        delta_avg += quantile_sum / (quantiles - 1) - set_value

    if max_quantiles > 1:
        delta_avg /= max_quantiles - 1
    else:
        delta_avg = 0
    return abs(delta_avg)


def precision(tp, fp, fn):
    if tp + fp > 0:
        return tp / (tp + fp)
    return 0


def recall(tp, fp, fn):
    if tp + fn > 0:
        return tp / (tp + fn)
    return 0


def fscore(tp, fp, fn):
    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    if p + r > 0:
        return 2 * (p * r) / (p + r)
    return 0


def confusion_matrix(hat_y, y, n_classes=None):
    hat_y = np.array(list(collapse(hat_y)))
    y = np.array(list(collapse(y)))

    if n_classes is None:
        classes = np.unique(np.union1d(hat_y, y))
        n_classes = len(classes)

    cnfm = np.zeros((n_classes, n_classes))
    for j in range(y.shape[0]):
        cnfm[y[j], hat_y[j]] += 1
    return cnfm


def scores_for_class(class_index, cnfm):
    tp = cnfm[class_index, class_index]
    fp = cnfm[:, class_index].sum() - tp
    fn = cnfm[class_index, :].sum() - tp
    tn = cnfm.sum() - tp - fp - fn

    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    f1 = fscore(tp, fp, fn)
    support = tp + tn
    return p, r, f1, support


def precision_recall_fscore_support(hat_y, y, labels=None):
    n_classes = len(labels) if labels else None
    cnfm = confusion_matrix(hat_y, y, n_classes)

    if n_classes is None:
        n_classes = cnfm.shape[0]

    scores = np.zeros((n_classes, 4))
    for class_id in range(n_classes):
        scores[class_id] = scores_for_class(class_id, cnfm)
    return scores.T.tolist()


def f1_product(hat_y, y):
    p, r, f1, s = precision_recall_fscore_support(hat_y, y)
    f1_mult = np.prod(f1)
    return f1_mult


def f1_scores(hat_y, y):
    """
    Return f1_bad, f1_ok and f1_product
    """
    p, r, f1, s = precision_recall_fscore_support(hat_y, y)
    f_mult = np.prod(f1)

    return (*f1, f_mult)
