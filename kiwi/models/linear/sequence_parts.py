class SequenceUnigramPart(object):
    """A part for unigrams (a single label at a word position)."""

    def __init__(self, index, label):
        self.label = label
        self.index = index


class SequenceBigramPart(object):
    """A part for bigrams (two labels at consecutive words position).
    Necessary for the model to be sequential."""

    def __init__(self, index, label, previous_label):
        self.label = label
        self.previous_label = previous_label
        self.index = index
