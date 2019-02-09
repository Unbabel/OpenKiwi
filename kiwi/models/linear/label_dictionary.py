# -*- coding: utf-8 -*-
"""This implements a dictionary of labels."""

import warnings


class LabelDictionary(dict):
    """This class implements a dictionary of labels. Labels as mapped to
    integers, and it is efficient to retrieve the label name from its
    integer representation, and vice-versa."""

    def __init__(self, label_names=None):
        dict.__init__(self)
        self.names = []
        if label_names is not None:
            for name in label_names:
                self.add(name)

    def add(self, name):
        """Add new label."""
        label_id = len(self.names)
        if name in self:
            warnings.warn('Ignoring duplicated label ' + name)
        self[name] = label_id
        self.names.append(name)
        return label_id

    def get_label_name(self, label_id):
        """Get label name from id."""
        return self.names[label_id]

    def get_label_id(self, name):
        """Get label id from name."""
        return self[name]

    def save(self, label_file):
        """Save labels to a file."""
        f = open(label_file, 'w')
        for name in self.names:
            f.write(name + '\n')
        f.close()

    def load(self, label_file):
        """Load labels from a file."""
        self.names = []
        self.clear()
        f = open(label_file)
        for line in f:
            name = line.rstrip('\n')
            self.add(name)
        f.close()
