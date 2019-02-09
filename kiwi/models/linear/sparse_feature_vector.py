"""This defines the class for defining sparse features in linear models."""

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

from .sparse_vector import SparseVector


class SparseFeatureVector(SparseVector):
    """A generic class for a sparse feature vector."""

    def __init__(
        self,
        save_to_cache=False,
        load_from_cache=False,
        cached_features_file=None,
    ):
        SparseVector.__init__(self)
        self.cached_features_file = cached_features_file
        self.save_to_cache = save_to_cache
        self.load_from_cache = load_from_cache

    def add_categorical_feature(self, name, value, allow_duplicates=False):
        """Add a categorical feature, represented internally as a binary
        feature."""
        fname = name + "=" + value
        assert allow_duplicates or fname not in self
        self[fname] = 1.0

    def add_binary_feature(self, name):
        """Add a binary feature."""
        if name in self:
            return
        self[name] = 1.0

    def add_numeric_feature(self, name, value):
        """Add a numeric feature."""
        self[name] = value

    def save_cached_features(self):
        """Save features to file."""
        self.cached_features_file.write(str(len(self)) + '\n')
        for key in self:
            self.cached_features_file.write(key + '\t' + str(self[key]) + '\n')

    def load_cached_features(self):
        """Load features from file."""
        num_features = int(self.cached_features_file.next())
        for i in range(num_features):
            key, value = (
                self.cached_features_file.next().rstrip('\n').split('\t')
            )
            self[key] = float(value)


class SparseBinaryFeatureVector(list):
    """A generic class for a sparse binary feature vector."""

    def __init__(
        self,
        feature_indices=None,
        save_to_cache=False,
        load_from_cache=False,
        cached_features_file=None,
    ):
        list.__init__(self)
        self.feature_indices = feature_indices
        self.cached_features_file = cached_features_file
        self.save_to_cache = save_to_cache
        self.load_from_cache = load_from_cache

    def add_categorical_feature(self, name, value):
        """Add a categorical feature, represented internally as a binary
        feature."""
        fname = name + "=" + value
        self.add_binary_feature(fname)

    def add_binary_feature(self, name):
        """Add a binary feature."""
        add = True
        index = self.feature_indices.get(name, -1)
        if index < 0:
            if not add:
                return
            else:
                index = self.feature_indices.add(name)
        self.append(index)

    def to_sparse_vector(self):
        """Convert to a SparseVector."""
        vector = SparseVector()
        for index in self:
            vector[index] = 1.0
        return vector

    def save_cached_features(self):
        """Save features to file."""
        self.cached_features_file.write(
            '\t'.join([str(key) for key in self]) + '\n'
        )

    def load_cached_features(self):
        """Load features from file."""
        self[:] = [
            int(key)
            for key in self.cached_features_file.next().rstrip('\n').split('\t')
        ]
