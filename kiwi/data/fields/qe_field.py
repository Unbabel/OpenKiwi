from collections.__init__ import Counter, OrderedDict
from itertools import chain

from torchtext import data

from kiwi.constants import PAD, START, STOP, UNALIGNED, UNK
from kiwi.data.vocabulary import Vocabulary


class QEField(data.Field):
    def __init__(
        self,
        unaligned_token=UNALIGNED,
        unk_token=UNK,
        pad_token=PAD,
        init_token=START,
        eos_token=STOP,
        **kwargs,
    ):
        kwargs.setdefault('batch_first', True)
        super().__init__(**kwargs)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.init_token = init_token
        self.eos_token = eos_token
        self.unaligned_token = unaligned_token
        self.vocab = None
        self.vocab_cls = Vocabulary

    def build_vocab(self, *args, **kwargs):
        """Add unaligned_token to the list of special symbols."""
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, data.Dataset):
                sources += [
                    getattr(arg, name)
                    for name, field in arg.fields.items()
                    if field is self
                ]
            else:
                sources.append(arg)
        for sample in sources:
            for x in sample:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(
            OrderedDict.fromkeys(
                tok
                for tok in [
                    self.unk_token,
                    self.pad_token,
                    self.init_token,
                    self.eos_token,
                    self.unaligned_token,
                ]
                if tok is not None
            )
        )
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)
