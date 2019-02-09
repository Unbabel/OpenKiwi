def tokenizer(sentence):
    """Implement your own tokenize procedure."""
    return sentence.strip().split()


def align_tokenizer(s):
    """Return a list of pair of integers for each sentence."""
    return [tuple(map(int, x.split('-'))) for x in s.strip().split()]


def align_reversed_tokenizer(s):
    """Return a list of pair of integers for each sentence."""
    return [tuple(map(int, x.split('-')))[::-1] for x in s.strip().split()]
