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
import pytest

from kiwi.metrics.functions import (
    confusion_matrix,
    f1_product,
    fscore,
    matthews_correlation_coefficient,
)


@pytest.fixture
def labels():
    n_class = 2
    y_gold = np.array(
        [
            np.array([1, 1, 0, 1]),
            np.array([1, 1, 0, 1, 1, 1, 1, 0]),
            np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 1]),
        ]
    )
    y_hat = np.array(
        [
            np.array([1, 1, 0, 0]),
            np.array([1, 1, 0, 1, 1, 1, 1, 0]),
            np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0]),
        ]
    )
    return y_gold, y_hat, n_class


def test_fscore(labels, atol):
    y_gold, y_hat, n_class = labels
    cnfm = confusion_matrix(y_hat, y_gold, n_class)
    f1_orig_prod_micro = f1_product(y_hat, y_gold)
    f1_prod_macro = 0
    f1_orig_prod_macro = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for ys_hat, ys_gold in zip(y_hat, y_gold):
        ctp = np.sum((ys_hat == 1) & (ys_gold == 1))
        ctn = np.sum((ys_hat == 0) & (ys_gold == 0))
        cfp = np.sum((ys_hat == 1) & (ys_gold == 0))
        cfn = np.sum((ys_hat == 0) & (ys_gold == 1))
        tp += ctp
        tn += ctn
        fp += cfp
        fn += cfn
        f_ok = fscore(ctp, cfp, cfn)
        f_bad = fscore(ctn, cfp, cfn)
        f1_prod_macro += f_ok * f_bad
        f1_orig_prod_macro += f1_product(ys_hat, ys_gold)

    assert tn == cnfm[0, 0]
    assert fp == cnfm[0, 1]
    assert fn == cnfm[1, 0]
    assert tp == cnfm[1, 1]

    f_ok = fscore(tp, fp, fn)
    f_bad = fscore(tn, fp, fn)
    f1_prod_micro = f_ok * f_bad
    f1_prod_macro = f1_prod_macro / y_gold.shape[0]
    f1_orig_prod_macro = f1_orig_prod_macro / y_gold.shape[0]

    np.testing.assert_allclose(f1_prod_micro, f1_orig_prod_micro, atol=atol)
    np.testing.assert_allclose(f1_prod_macro, f1_orig_prod_macro, atol=atol)


def test_matthews(labels, atol):
    y_gold, y_hat, n_class = labels
    matthews = matthews_correlation_coefficient(y_hat, y_gold)
    np.testing.assert_allclose(matthews, 0.70082555, atol=atol)
    n1 = matthews_correlation_coefficient(y_hat, 1 - y_gold)
    n2 = matthews_correlation_coefficient(1 - y_hat, y_gold)
    nn = matthews_correlation_coefficient(1 - y_hat, 1 - y_gold)
    assert n1 == n2
    assert nn == matthews
    assert n1 == -matthews


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
