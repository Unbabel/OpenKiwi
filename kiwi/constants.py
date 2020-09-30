#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2020 Unbabel <openkiwi@unbabel.com>
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

SOURCE = 'source'
TARGET = 'target'
PE = 'pe'
TARGET_TAGS = 'target_tags'
SOURCE_TAGS = 'source_tags'
GAP_TAGS = 'gap_tags'
TARGETGAPS_TAGS = 'targetgaps_tags'

SOURCE_LOGITS = 'source_logits'
TARGET_LOGITS = 'target_logits'
TARGET_SENTENCE = 'target_sentence'
PE_LOGITS = 'pe_logits'

SENTENCE_SCORES = 'sentence_scores'
BINARY = 'binary'

ALIGNMENTS = 'alignments'
SOURCE_POS = 'source_pos'
TARGET_POS = 'target_pos'

# Constants for model output names
LOSS = 'loss'

# Standard Names for saving files
VOCAB = 'vocab'
CONFIG = 'config'
STATE_DICT = 'state_dict'
