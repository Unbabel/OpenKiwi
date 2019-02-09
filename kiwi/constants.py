# lowercased special tokens
UNK = '<unk>'
PAD = '<pad>'
START = '<bos>'
STOP = '<eos>'
UNALIGNED = '<unaligned>'

# special tokens id (don't edit this order)
# FIXME: avoid using these IDs since we don't really make sure they correspond
# to the above tokens
UNK_ID = 0
PAD_ID = 1
START_ID = 2
STOP_ID = 3
UNALIGNED_ID = 4

PAD_TAGS_ID = 2
# binary labels
OK = 'OK'
BAD = 'BAD'
OK_ID = 0
BAD_ID = 1
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
