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
import math
from typing import Any, Tuple

import numpy as np
from more_itertools import collapse


def mean_absolute_error(y, y_hat):
    return np.mean(np.absolute(y_hat - y))


def mean_squared_error(y, y_hat):
    return np.square(np.subtract(y, y_hat)).mean()


def delta_average(y_true, y_rank) -> float:
    """Calculate the DeltaAvg score.

    This is a much faster version than the Perl one provided in the WMT QE task 1.

    References: could not find any.

    Author: Fabio Kepler (contributed to MARMOT).

    Arguments:
        y_true: array of reference score (not rank) of each segment.
        y_rank: array of rank of each segment.

    Return:
         the absolute delta average score.
    """
    sorted_ranked_indexes = np.argsort(y_rank)
    y_length = len(sorted_ranked_indexes)

    delta_avg = 0
    max_quantiles = y_length // 2
    set_value = np.sum(y_true[sorted_ranked_indexes[np.arange(y_length)]]) / y_length
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
    # tn = cnfm.sum() - tp - fp - fn

    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    f1 = fscore(tp, fp, fn)
    support = tp + fn
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


def f1_scores(hat_y, y) -> Tuple[Any, np.ndarray]:
    """Compute and return f1 for each class and the f1_product."""
    p, r, f1, s = precision_recall_fscore_support(hat_y, y)
    f_mult = np.prod(f1)

    return (*f1, f_mult)


def matthews_correlation_coefficient(hat_y, y):
    """Compute Matthews Correlation Coefficient.

    Arguments:
        hat_y: list of np array of predicted binary labels.
        y: list of np array of true binary labels.

    Return:
        the Matthews correlation coefficient of hat_y and y.
    """

    cnfm = confusion_matrix(hat_y, y, 2)
    tp = cnfm[0][0]
    tn = cnfm[1][1]
    fp = cnfm[1][0]
    fn = cnfm[0][1]
    class_p = tp + fn
    class_n = tn + fp
    pred_p = tp + fp
    pred_n = tn + fn
    normalizer = class_p * class_n * pred_p * pred_n

    if normalizer:
        return ((tp * tn) - (fp * fn)) / math.sqrt(normalizer)
    else:
        return 0
