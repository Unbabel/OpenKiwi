from collections import Counter

from torchtext.data import Field


class SequenceLabelsField(Field):
    """Sequence of Labels.
    """

    def __init__(self, classes, *args, **kwargs):
        self.classes = classes
        self.vocab = None
        super().__init__(*args, **kwargs)

    def build_vocab(self, *args, **kwargs):
        specials = self.classes + [
            self.pad_token,
            self.init_token,
            self.eos_token,
        ]
        self.vocab = self.vocab_cls(Counter(), specials=specials, **kwargs)
