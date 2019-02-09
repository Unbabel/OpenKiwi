"""This implements a linear model."""
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


class LinearModel(object):
    """ An abstract linear model."""

    def __init__(self):
        self.use_average = True
        self.weights = SparseVector()
        self.averaged_weights = SparseVector()

    def clear(self):
        """Clear all weights."""
        self.weights.clear()
        self.averaged_weights.clear()

    def finalize(self, t):
        """Finalize by setting the weights as the running average.
        This is a no-op if use_average=False."""
        if self.use_average:
            self.averaged_weights.scale(1.0 / float(t))
            self.weights.add(self.averaged_weights)

    def compute_score(self, features):
        """Compute a score by taking the inner product with a feature
        vector."""
        score = features.dot_product(self.weights)
        return score

    def compute_score_binary_features(self, binary_features):
        """Compute a score by taking the inner product with a binary
        feature vector."""
        score = 0.0
        for f in binary_features:
            if f in self.weights:
                score += self.weights[f]
        return score

    def make_gradient_step(self, features, eta, t, gradient):
        """Make a gradient step with stepsize eta."""
        self.weights.add(features, -eta * gradient)
        if self.use_average:
            self.averaged_weights.add(features, eta * float(t) * gradient)

    def save(self, model_file, average=False, feature_indices=None):
        """Save the model to a file."""
        f = open(model_file, 'w')
        if feature_indices is not None:
            w = SparseVector()
            for index in self.weights:
                w[feature_indices.get_label_name(index)] = self.weights[index]
            w.save(f)
        else:
            self.weights.save(f)
        f.close()
        if average:
            f = open(model_file + '_average', 'w')
            if feature_indices is not None:
                w = SparseVector()
                for index in self.averaged_weights:
                    w[
                        feature_indices.get_label_name(index)
                    ] = self.averaged_weights[index]
                w.save(f)
            else:
                self.averaged_weights.save(f)
            f.close()

    def load(self, model_file, average=False, feature_indices=None):
        """Load the model from a file."""
        f = open(model_file, 'r')
        if feature_indices is not None:
            w = SparseVector()
            w.load(f)
            for key in w:
                index = feature_indices.add(key)
                self.weights[index] = w[key]
        else:
            self.weights.load(f)
        f.close()
        if average:
            f = open(model_file + '_average', 'r')
            if feature_indices is not None:
                w = SparseVector()
                w.load(f)
                for key in w:
                    index = feature_indices.get_label_id(key)
                    self.averaged_weights[index] = w[key]
            else:
                self.averaged_weights.load(f)
            f.close()

    def write_fnames(self, fnames_file, fnames):
        """Write file mapping from integers to feature descriptions."""
        f = open(fnames_file, 'w')
        for fid, fname in enumerate(fnames):
            f.write(str(1 + fid) + ' ' + fname + '\n')
        f.close()

    def read_fnames(self, fnames_file):
        """Read file mapping from integers to feature descriptions."""
        assert False, 'This is not being called'
        fids = {}
        f = open(fnames_file)
        maxfid = -1
        for line in f:
            line = line.rstrip('\n')
            fields = line.split(' ')
            fid = int(fields[0])
            fname = fields[1]
            fids[fname] = fid
            if fid > maxfid:
                maxfid = fid
        fnames = [''] * maxfid
        for fname, fid in fids.iteritems():
            fnames[fid - 1] = fname
        f.close()
        return fnames, fids
