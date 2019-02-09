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

import string


class LinearWordQETokenFeatures(object):
    def __init__(
        self,
        stacked_features=None,
        source_token_count=-1,
        target_token_count=-1,
        source_target_token_count_ratio=0.0,
        token='',
        left_context='',
        right_context='',
        first_aligned_token='',
        left_alignment='',
        right_alignment='',
        is_stopword=False,
        is_punctuation=False,
        is_proper_noun=False,
        is_digit=False,
        highest_order_ngram_left=-1,
        highest_order_ngram_right=-1,
        backoff_behavior_left=0.0,
        backoff_behavior_middle=0.0,
        backoff_behavior_right=0.0,
        source_highest_order_ngram_left=-1,
        source_highest_order_ngram_right=-1,
        pseudo_reference=False,
        target_pos='',
        target_morph='',
        target_head=-1,
        target_deprel='',
        aligned_source_pos_list='',
        polysemy_count_source=0,
        polysemy_count_target=0,
    ):
        self.stacked_features = (
            stacked_features if stacked_features is not None else []
        )
        self.source_token_count = source_token_count  # Not used.
        self.target_token_count = target_token_count  # Not used.
        # Not used.
        self.source_target_token_count_ratio = source_target_token_count_ratio
        self.token = token
        self.left_context = left_context
        self.right_context = right_context
        self.first_aligned_token = first_aligned_token
        self.left_alignment = left_alignment
        self.right_alignment = right_alignment
        self.is_stopword = is_stopword  # Not used (at least for En-De).
        self.is_punctuation = is_punctuation
        self.is_proper_noun = is_proper_noun  # Not used (at least for En-De).
        self.is_digit = is_digit
        self.highest_order_ngram_left = highest_order_ngram_left
        self.highest_order_ngram_right = highest_order_ngram_right
        self.backoff_behavior_left = backoff_behavior_left  # Not used.
        self.backoff_behavior_middle = backoff_behavior_middle  # Not used.
        self.backoff_behavior_right = backoff_behavior_right  # Not used.
        self.source_highest_order_ngram_left = source_highest_order_ngram_left
        self.source_highest_order_ngram_right = source_highest_order_ngram_right
        self.pseudo_reference = pseudo_reference  # Not used in the WMT16+ data.
        self.target_pos = target_pos
        self.target_morph = target_morph  # Not used.
        self.target_head = target_head
        self.target_deprel = target_deprel
        self.aligned_source_pos_list = aligned_source_pos_list
        self.polysemy_count_source = polysemy_count_source  # Not used.
        self.polysemy_count_target = polysemy_count_target  # Not used.


class LinearWordQESentence:
    """Represents a sentence (word features and their labels)."""

    @staticmethod
    def create_stop_symbol():
        """Generates dummy features for a stop symbol."""
        return LinearWordQETokenFeatures(
            token='__STOP__',
            left_context='__STOP__',
            right_context='__STOP__',
            first_aligned_token='__STOP__',
            left_alignment='__STOP__',
            right_alignment='__STOP__',
            target_pos='__STOP__',
            aligned_source_pos_list='__STOP__',
            target_morph='__STOP__',
            target_deprel='__STOP__',
        )

    def __init__(self):
        self.sentence_word_features = []
        self.sentence_word_labels = []

    def num_words(self):
        """Returns the number of words of the sentence."""
        return len(self.sentence_word_features)

    def create_from_sentence_pair(
        self,
        source_words,
        target_words,
        alignments,
        source_pos_tags=None,
        target_pos_tags=None,
        target_parse_heads=None,
        target_parse_relations=None,
        target_ngram_left=None,
        target_ngram_right=None,
        target_stacked_features=None,
        labels=None,
    ):
        """Creates an instance from source/target token and alignment
        information."""
        self.sentence_word_features = []
        aligned_source_words = [[] for _ in target_words]
        for source, target in alignments:
            aligned_source_words[target].append(source)
        aligned_source_words = [
            sorted(aligned) for aligned in aligned_source_words
        ]
        if source_pos_tags is None:
            source_pos_tags = ['' for _ in source_words]
        if target_pos_tags is None:
            target_pos_tags = ['' for _ in target_words]
        if target_parse_heads is None:
            target_parse_heads = [-1 for _ in target_words]
        if target_parse_relations is None:
            target_parse_relations = ['' for _ in target_words]
        if target_ngram_left is None:
            target_ngram_left = [-1 for _ in target_words]
        if target_ngram_right is None:
            target_ngram_right = [-1 for _ in target_words]
        if target_stacked_features is None:
            target_stacked_features = ['' for _ in target_words]
        if labels is not None:
            if len(labels) != len(target_words):
                # WMT18 format with labels for the gaps.
                assert len(labels) == 2 * len(target_words) + 1
                labels = labels[1::2]
        for i in range(len(target_words)):
            word = target_words[i]
            tag = target_pos_tags[i]
            parse_head = int(target_parse_heads[i])  # TODO: don't cast here.
            parse_relation = target_parse_relations[i]
            ngram_left = int(target_ngram_left[i])  # TODO: don't cast here.
            ngram_right = int(target_ngram_right[i])  # TODO: don't cast here.
            if not target_stacked_features[i]:
                stacked_features = None
            else:
                stacked_features = [
                    float(p) for p in target_stacked_features[i].split('|')
                ]
            if i == 0:
                previous_word = '<s>'
                # previous_tag = '<s>'
            else:
                previous_word = target_words[i - 1]
                # previous_tag = target_pos_tags[i-1]
            if i == len(target_words) - 1:
                next_word = '</s>'
                # next_tag = '</s>'
            else:
                next_word = target_words[i + 1]
                # next_tag = target_pos_tags[i+1]
            if len(aligned_source_words[i]) == 0:
                source_word = '__unaligned__'
                previous_source_word = '__unaligned__'
                next_source_word = '__unaligned__'
                source_tag = '__unaligned__'
                # previous_source_tag = '__unaligned__'
                # next_source_tag = '__unaligned__'
            else:
                # Concatenate all source words in order of appearance.
                # The previous word is the one before the first aligned
                # source word; the next word is the one after the last
                # aligned word.
                source_word = '|'.join(
                    [source_words[j] for j in aligned_source_words[i]]
                )
                source_tag = '|'.join(
                    [source_pos_tags[j] for j in aligned_source_words[i]]
                )
                j = aligned_source_words[i][0]
                if j == 0:
                    previous_source_word = "<s>"
                    # previous_source_tag = "<s>"
                else:
                    previous_source_word = source_words[j - 1]
                    # previous_source_tag = source_pos_tags[j-1]
                if j == len(source_words) - 1:
                    next_source_word = "</s>"
                    # next_source_tag = "</s>"
                else:
                    next_source_word = source_words[j + 1]
                    # next_source_tag = source_pos_tags[j+1]

            word_features = LinearWordQETokenFeatures(
                stacked_features=stacked_features,
                source_token_count=len(source_words),
                target_token_count=len(target_words),
                source_target_token_count_ratio=float(len(source_words))
                / len(target_words),
                token=word,
                is_punctuation=all([c in string.punctuation for c in word]),
                is_digit=word.isdigit(),
                target_pos=tag,
                left_context=previous_word,
                right_context=next_word,
                first_aligned_token=source_word,
                aligned_source_pos_list=source_tag,
                left_alignment=previous_source_word,
                right_alignment=next_source_word,
                target_head=parse_head,
                target_deprel=parse_relation,
                highest_order_ngram_left=ngram_left,
                highest_order_ngram_right=ngram_right,
            )

            self.sentence_word_features.append(word_features)
            self.sentence_word_labels.append(
                labels[i] if labels is not None else ''
            )
