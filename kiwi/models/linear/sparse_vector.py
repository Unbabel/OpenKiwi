# -*- coding: utf-8 -*-
"""This defines a generic class for sparse vectors."""

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

import math


class SparseVector(dict):
    """Implementation of a sparse vector using a dictionary."""

    def __init__(self):
        dict.__init__(self)

    def copy(self):
        """Returns a copy of the current vector."""
        vector = SparseVector()
        for key in self:
            vector[key] = self[key]
        return vector

    def as_string(self):
        """Returns a string representation."""
        s = ''
        for key in self:
            s += key + ':' + str(self[key]) + ' '
        return s

    def save(self, f):
        """Save vector to file."""
        for key in self:
            f.write(str(key) + '\t' + str(self[key]) + '\n')

    def load(self, f, dtype=str):
        """Load vector from file."""
        self.clear()
        for line in f:
            fields = line.split('\t')
            key = fields[0]
            value = float(fields[1])
            self[dtype(key)] = value

    def add(self, vector, scalar=1.0):
        """ Adds this vector and a given vector."""
        for key in vector:
            if key in self:
                self[key] += scalar * vector[key]
            else:
                self[key] = scalar * vector[key]

    def scale(self, scalar):
        """Scales this vector by a scale factor."""
        for key in self:
            self[key] *= scalar

    def add_constant(self, scalar):
        """Adds a constant to each element of the vector."""
        for key in self:
            self[key] += scalar

    def squared_norm(self):
        """Computes the squared norm of the vector."""
        return self.dot_product(self)

    def dot_product(self, vector):
        """ Computes the dot product with a given vector.
        Note: this iterates through the self vector, so it may be inefficient
        if the number of nonzeros in self is much larger than the number of
        nonzeros in vector. Hence the function reverts to
        vector.dot_product(self) if that is beneficial."""
        if len(self) > len(vector):
            return vector.dot_product(self)
        value = 0.0
        for key in self:
            if key in vector:
                value += self[key] * vector[key]
        return value

    def normalize(self):
        """ Normalize the vector. Note: if the norm is zero, do nothing."""
        norm = 0.0
        for key in self:
            value = self[key]
            norm += value * value
        norm = math.sqrt(norm)
        if norm > 0.0:
            for key in self:
                self[key] /= norm
