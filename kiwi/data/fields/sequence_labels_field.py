from collections import Counter
import copy

from torchtext.data import Field
from kiwi.data.vocabulary import Vocabulary
from kiwi import constants as const


class SequenceLabelsField(Field):
    """Sequence of Labels.
    """

    def __init__(self, classes=None, vocab_cls=Vocabulary, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if classes is None:
            classes = const.LABELS
        self.classes = classes
        self.vocab = None
        self.vocab_cls = vocab_cls

    def build_vocab(self, *args, **kwargs):
        specials = copy.copy(self.classes)
        if self.pad_token:
            specials.append(self.pad_token)
        # cnt = Counter({class_name: 1 for class_name in self.classes})
        self.vocab = self.vocab_cls(Counter(), specials=specials, **kwargs)
