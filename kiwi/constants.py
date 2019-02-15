# lowercased special tokens
UNK = '<unk>'
PAD = '<pad>'
START = '<bos>'
STOP = '<eos>'
UNALIGNED = '<unaligned>'

# binary labels
OK = 'OK'
BAD = 'BAD'
LABELS = [OK, BAD]

# this should be removed in the future
# fields
SOURCE = 'source'
TARGET = 'target'
PE = 'pe'
TARGET_TAGS = 'tags'
SOURCE_TAGS = 'source_tags'
GAP_TAGS = 'gap_tags'

TAGS = [TARGET_TAGS, SOURCE_TAGS, GAP_TAGS]

SENTENCE_SCORES = 'sentence_scores'
BINARY = 'binary'

TARGETS = [SENTENCE_SCORES, BINARY] + TAGS

ALIGNMENTS = 'alignments'
SOURCE_POS = 'source_pos'
TARGET_POS = 'target_pos'
TARGET_PARSE_HEADS = 'target_parse_heads'
TARGET_PARSE_RELATIONS = 'target_parse_relations'
TARGET_NGRAM_LEFT = 'target_ngram_left'
TARGET_NGRAM_RIGHT = 'target_ngram_right'
TARGET_STACKED = 'target_stacked'

# Constants for model output names
SENT_SIGMA = 'sentence_sigma'
LOSS = 'loss'
PREQEFV = 'PreQEFV'
POSTQEFV = 'PostQEFV'

# Standard Names for saving files
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
EVAL = 'eval'
VOCAB = 'vocab'
CONFIG = 'config'
STATE_DICT = 'state_dict'
VOCAB_FILE = 'vocab.torch'
MODEL_FILE = 'model.torch'
DATAFILE = 'dataset.torch'
OPTIMIZER = 'optim.torch'
BEST_MODEL_FILE = 'best_model.torch'
TRAINER = 'trainer.torch'
EPOCH = 'epoch'
