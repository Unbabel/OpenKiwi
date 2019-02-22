# -*- coding: utf-8 -*-
"""Several utility functions."""


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


def nearly_eq_tol(a, b, tol):
    """Checks if two numbers are equal up to a tolerance."""
    return (a - b) * (a - b) <= tol


def nearly_binary_tol(a, tol):
    """Checks if a number is binary up to a tolerance."""
    return nearly_eq_tol(a, 0.0, tol) or nearly_eq_tol(a, 1.0, tol)


def nearly_zero_tol(a, tol):
    """Checks if a number is zero up to a tolerance."""
    return (a <= tol) and (a >= -tol)
