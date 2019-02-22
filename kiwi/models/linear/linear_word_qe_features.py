"""A class for handling features for word-level quality estimation."""

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

import numpy as np

from kiwi.models.linear.linear_word_qe_sentence import LinearWordQESentence

from .sparse_feature_vector import SparseFeatureVector


def quantize(value, bins_down):
    """Quantize a numeric feature into bins.
    Example: bins = [50, 40, 30, 25, 20, 18, 16, 14, 12, 10]."""
    bin_up = np.inf
    for bin_down in bins_down:
        if bin_down < value <= bin_up:
            bin_value = bin_down
            return bin_value
        bin_up = bin_down
    return value


class LinearWordQEFeatures(SparseFeatureVector):
    """This class implements a feature vector for word-level quality
    estimation."""

    def __init__(
        self,
        use_basic_features_only=True,
        use_simple_bigram_features=True,
        use_parse_features=False,
        use_stacked_features=False,
        save_to_cache=False,
        load_from_cache=False,
        cached_features_file=None,
    ):
        SparseFeatureVector.__init__(
            self, save_to_cache, load_from_cache, cached_features_file
        )
        self.use_basic_features_only = use_basic_features_only
        # True for using only a single bigram indicator feature.
        self.use_simple_bigram_features = use_simple_bigram_features
        self.use_parse_features = use_parse_features
        self.use_stacked_features = use_stacked_features
        self.use_client_features = False

    def get_siblings(self, sentence_word_features, index):
        if index < 0 or index >= len(sentence_word_features):
            info = None
        else:
            info = sentence_word_features[index]
        if info is not None:
            siblings = [
                k
                for k in range(len(sentence_word_features))
                if sentence_word_features[k].target_head == info.target_head
            ]
            left_siblings = [k for k in siblings if k < index]
            right_siblings = [k for k in siblings if k > index]
            if len(left_siblings) > 0:
                left_sibling = max(left_siblings)
            else:
                left_sibling = -1
            if len(right_siblings) > 0:
                right_sibling = min(right_siblings)
            else:
                right_sibling = -1
        else:
            left_sibling = -2
            right_sibling = -2

        if left_sibling >= 0:
            left_sibling_info = sentence_word_features[left_sibling]
            left_sibling_token = left_sibling_info.token
            left_sibling_pos = left_sibling_info.target_pos
        elif left_sibling == -1:
            left_sibling_token = '__ROOT__'
            left_sibling_pos = '__ROOT__'
        else:
            left_sibling_info = None
            left_sibling_token = '__START__'
            left_sibling_pos = '__START__'
        if right_sibling >= 0:
            right_sibling_info = sentence_word_features[right_sibling]
            right_sibling_token = right_sibling_info.token
            right_sibling_pos = right_sibling_info.target_pos
        elif right_sibling == -1:
            right_sibling_info = None
            right_sibling_token = '__ROOT__'
            right_sibling_pos = '__ROOT__'
        else:
            right_sibling_info = None
            right_sibling_token = '__START__'
            right_sibling_pos = '__START__'

        return (
            left_sibling_token,
            left_sibling_pos,
            right_sibling_token,
            right_sibling_pos,
        )

    def get_head(self, sentence_word_features, index):
        if index < 0 or index >= len(sentence_word_features):
            info = None
        else:
            info = sentence_word_features[index]
        if info is not None:
            head_index = info.target_head - 1
        else:
            head_index = -2
        if head_index >= 0:
            head_info = sentence_word_features[head_index]
            head_token = head_info.token
            head_pos = head_info.target_pos
            head_morph = head_info.target_morph
        elif head_index == -1:
            head_info = None
            head_token = '__ROOT__'
            head_pos = '__ROOT__'
            head_morph = '__ROOT__'
        else:
            head_info = None
            head_token = '__START__'
            head_pos = '__START__'
            head_morph = '__START__'

        return head_index, head_token, head_pos, head_morph

    def compute_unigram_features(self, sentence_word_features, part):
        """Compute unigram features (depending only on a single label)."""
        if self.load_from_cache:
            self.load_cached_features()
            return

        index = part.index

        ignore_source = False
        only_basic_features = self.use_basic_features_only
        use_client_features = self.use_client_features
        use_parse_features = self.use_parse_features
        use_stacked_features = self.use_stacked_features
        use_bias = True
        use_language_model = True
        use_binary_features = False
        if use_parse_features:
            use_split_morphs = False
            use_morph_features = False
            use_deprel_features = True
            use_head_features = True
            use_grandparent_features = True
            use_sibling_features = True
        else:
            use_split_morphs = False
            use_morph_features = False
            use_deprel_features = False
            use_head_features = False
            use_grandparent_features = False
            use_sibling_features = False
        use_unuseful_shared_task_features = False

        info = sentence_word_features[index]
        if use_client_features:
            labels = [str(part.label), info.client_name + '_' + str(part.label)]
        else:
            labels = [str(part.label)]

        for label in labels:
            if use_bias:
                self.add_binary_feature('BIAS_%s' % label)

            if use_unuseful_shared_task_features:
                self.add_binary_feature(
                    'F0=%d_%s'
                    % (
                        quantize(info.source_token_count, [40, 30, 20, 10]),
                        label,
                    )
                )
                self.add_binary_feature(
                    'F1=%d_%s'
                    % (
                        quantize(info.target_token_count, [40, 30, 20, 10]),
                        label,
                    )
                )
                self.add_binary_feature(
                    'F2=%f_%s'
                    % (
                        quantize(
                            info.source_target_token_count_ratio, [5.0, 2.0]
                        ),
                        label,
                    )
                )

            self.add_binary_feature('F3=%s_%s' % (info.token, label))
            self.add_binary_feature('F4=%s_%s' % (info.left_context, label))
            self.add_binary_feature('F5=%s_%s' % (info.right_context, label))

            if not ignore_source:
                self.add_binary_feature(
                    'F6=%s_%s' % (info.first_aligned_token, label)
                )
                self.add_binary_feature(
                    'F7=%s_%s' % (info.left_alignment, label)
                )
                self.add_binary_feature(
                    'F8=%s_%s' % (info.right_alignment, label)
                )

            if use_binary_features and not only_basic_features:
                # Ablated for German WMT16 (the provided stoplist is wrong).
                # self.add_binary_feature(
                #     'F9=%d_%s' % (int(info.is_stopword), label))
                self.add_binary_feature(
                    'F10=%d_%s' % (int(info.is_punctuation), label)
                )
                # Ablated for German (capitalized words are nouns)
                # self.add_binary_feature(
                #     'F11=%d_%s' % (int(info.is_proper_noun), label))
                self.add_binary_feature(
                    'F12=%d_%s' % (int(info.is_digit), label)
                )

            if use_language_model and not only_basic_features:
                self.add_binary_feature(
                    'F13=%d_%s' % (info.highest_order_ngram_left, label)
                )
                self.add_binary_feature(
                    'F14=%d_%s' % (info.highest_order_ngram_right, label)
                )

            # if use_language_model and not only_basic_features:
            #     self.add_binary_feature(
            #         'F15=%d_%s' % (info.backoff_behavior_left, label))
            #     self.add_binary_feature(
            #         'F16=%d_%s' % (info.backoff_behavior_middle, label))
            #     self.add_binary_feature(
            #         'F17=%d_%s' % (info.backoff_behavior_right, label))

            if use_language_model and not only_basic_features:
                self.add_binary_feature(
                    'F18=%d_%s' % (info.source_highest_order_ngram_left, label)
                )
                self.add_binary_feature(
                    'F19=%d_%s' % (info.source_highest_order_ngram_right, label)
                )
                self.add_binary_feature(
                    'F20=%d_%s' % (int(info.pseudo_reference), label)
                )

            if not only_basic_features:
                self.add_binary_feature('F21=%s_%s' % (info.target_pos, label))
                self.add_binary_feature(
                    'F22=%s_%s' % (info.aligned_source_pos_list, label)
                )

            if use_unuseful_shared_task_features:
                self.add_binary_feature(
                    'F23=%d_%s' % (info.polysemy_count_source, label)
                )
                self.add_binary_feature(
                    'F24=%d_%s' % (info.polysemy_count_target, label)
                )

            # QUETCH linear model conjoined features.
            self.add_binary_feature(
                'G0=%s_%s_%s' % (info.token, info.left_context, label)
            )
            self.add_binary_feature(
                'G1=%s_%s_%s' % (info.token, info.right_context, label)
            )
            if not ignore_source:
                self.add_binary_feature(
                    'G2=%s_%s_%s'
                    % (info.token, info.first_aligned_token, label)
                )
            if not only_basic_features:
                self.add_binary_feature(
                    'G3=%s_%s_%s'
                    % (info.target_pos, info.aligned_source_pos_list, label)
                )

            # Parse features.
            if use_parse_features:
                head_index, head_token, head_pos, head_morph = self.get_head(
                    sentence_word_features, index
                )
                head_on_left = True  # (head_index <= index)

                if head_index >= 0:
                    _, grandparent_token, grandparent_pos, _ = self.get_head(
                        sentence_word_features, head_index
                    )
                else:
                    grandparent_token, grandparent_pos = head_token, head_pos
                grandparent_on_left = True  # (grandparent_index <= index)

                left_sibling_token, left_sibling_pos, right_sibling_token, right_sibling_pos = self.get_siblings(  # NOQA
                    sentence_word_features, index
                )

                if use_deprel_features:
                    self.add_binary_feature(
                        'H0=%s_%s' % (info.target_deprel, label)
                    )
                    self.add_binary_feature(
                        'H1=%s_%s_%s' % (info.token, info.target_deprel, label)
                    )

                if use_head_features:
                    # self.add_binary_feature(
                    #     'H2=%s_%s_%s' % (info.target_pos, head_pos, label))
                    # self.add_binary_feature(
                    #     'H3=%s_%s_%s' % (info.token, head_token, label))
                    self.add_binary_feature(
                        'H2=%s_%s_%d_%s'
                        % (info.target_pos, head_pos, int(head_on_left), label)
                    )
                    self.add_binary_feature(
                        'H3=%s_%s_%d_%s'
                        % (info.token, head_token, int(head_on_left), label)
                    )
                    self.add_binary_feature(
                        'H3a=%s_%s_%d_%s'
                        % (info.token, head_pos, int(head_on_left), label)
                    )
                    self.add_binary_feature(
                        'H3b=%s_%s_%d_%s'
                        % (
                            info.target_pos,
                            head_token,
                            int(head_on_left),
                            label,
                        )
                    )

                if use_morph_features:
                    self.add_binary_feature(
                        'H4=%s_%s' % (info.target_morph, label)
                    )
                    self.add_binary_feature(
                        'H5=%s_%s_%s' % (info.target_morph, head_morph, label)
                    )

                if use_split_morphs:
                    all_morphs = info.target_morph.split('|')
                    all_head_morphs = head_morph.split('|')
                    for m in all_morphs:
                        self.add_binary_feature('H6=%s_%s' % (m, label))
                        for hm in all_head_morphs:
                            self.add_binary_feature(
                                'H7=%s_%s_%s' % (m, hm, label)
                            )

                if use_sibling_features:
                    self.add_binary_feature(
                        'H8=%s_%s_%s'
                        % (info.target_pos, left_sibling_pos, label)
                    )
                    self.add_binary_feature(
                        'H9=%s_%s_%s' % (info.token, left_sibling_token, label)
                    )
                    self.add_binary_feature(
                        'H10=%s_%s_%s'
                        % (info.target_pos, right_sibling_pos, label)
                    )
                    self.add_binary_feature(
                        'H11=%s_%s_%s'
                        % (info.token, right_sibling_token, label)
                    )

                if use_grandparent_features:
                    self.add_binary_feature(
                        'H12=%s_%s_%d_%s'
                        % (
                            info.target_pos,
                            grandparent_pos,
                            int(grandparent_on_left),
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'H13=%s_%s_%d_%s'
                        % (
                            info.token,
                            grandparent_token,
                            int(grandparent_on_left),
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'H14=%s_%s_%s_%d_%s'
                        % (
                            info.target_pos,
                            head_pos,
                            grandparent_pos,
                            int(grandparent_on_left),
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'H15=%s_%s_%s_%d_%s'
                        % (
                            info.token,
                            head_pos,
                            grandparent_token,
                            int(grandparent_on_left),
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'H16=%s_%s_%s_%d_%s'
                        % (
                            info.token,
                            head_token,
                            grandparent_pos,
                            int(grandparent_on_left),
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'H17=%s_%s_%s_%d_%s'
                        % (
                            info.target_pos,
                            head_token,
                            grandparent_token,
                            int(grandparent_on_left),
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'H18=%s_%s_%s_%d_%s'
                        % (
                            info.target_pos,
                            head_pos,
                            grandparent_token,
                            int(grandparent_on_left),
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'H19=%s_%s_%s_%d_%s'
                        % (
                            info.target_pos,
                            head_token,
                            grandparent_pos,
                            int(grandparent_on_left),
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'H20=%s_%s_%s_%d_%s'
                        % (
                            info.token,
                            head_pos,
                            grandparent_pos,
                            int(grandparent_on_left),
                            label,
                        )
                    )

            if use_stacked_features:
                if len(info.stacked_features) > 0:
                    for i, value in enumerate(info.stacked_features):
                        self.add_numeric_feature('S%d_%s' % (i, label), value)

        if self.save_to_cache:
            self.save_cached_features()
            return

    def compute_bigram_features(self, sentence_word_features, part):
        """Compute bigram features (that depend on consecutive labels)."""
        if self.load_from_cache:
            self.load_cached_features()
            return

        index = part.index
        label = part.label
        previous_label = part.previous_label

        ignore_source = False
        only_basic_features = self.use_basic_features_only
        use_client_features = self.use_client_features
        use_parse_features = self.use_parse_features
        use_stacked_features = self.use_stacked_features  # False
        use_bias = True
        # True for using only a single bigram indicator feature.
        use_only_bias = self.use_simple_bigram_features
        use_language_model = True
        use_binary_features = False
        use_trigram_features = True

        if use_parse_features:
            use_split_morphs = False
            use_morph_features = False
            use_deprel_features = True
            use_head_features = False
            use_sibling_features = False
        else:
            use_split_morphs = False
            use_morph_features = False
            use_deprel_features = False
            use_head_features = False
            use_sibling_features = False

        if index < len(sentence_word_features):
            info = sentence_word_features[index]
        else:
            info = LinearWordQESentence.create_stop_symbol()

        if index > 0:
            info_previous = sentence_word_features[index - 1]
        else:
            info_previous = LinearWordQESentence.create_stop_symbol()

        bigram_label = str(previous_label) + '_' + str(label)
        if use_client_features:
            labels = [bigram_label, info.client_name + '_' + bigram_label]
        else:
            labels = [bigram_label]

        for label in labels:
            if use_bias:
                self.add_binary_feature('B1=%s' % label)

            if use_only_bias:
                continue

            self.add_binary_feature('B2=%s_%s' % (info.token, label))
            self.add_binary_feature('B3=%s_%s' % (info_previous.token, label))
            self.add_binary_feature('B4=%s_%s' % (info.right_context, label))
            self.add_binary_feature(
                'B5=%s_%s' % (info_previous.left_context, label)
            )

            if not ignore_source:
                self.add_binary_feature(
                    'B6=%s_%s' % (info.first_aligned_token, label)
                )
                self.add_binary_feature(
                    'B7=%s_%s' % (info.left_alignment, label)
                )
                self.add_binary_feature(
                    'B8=%s_%s' % (info.right_alignment, label)
                )
                self.add_binary_feature(
                    'B9=%s_%s' % (info_previous.first_aligned_token, label)
                )
                self.add_binary_feature(
                    'B10=%s_%s' % (info_previous.left_alignment, label)
                )
                self.add_binary_feature(
                    'B11=%s_%s' % (info_previous.right_alignment, label)
                )

            if use_binary_features and not only_basic_features:
                # Ablated for German WMT16 (the provided stoplist is wrong).
                # self.add_binary_feature(
                #     'B12=%d_%s' % (int(info.is_stopword), label))
                # self.add_binary_feature(
                #     'B13=%d_%s' % (int(info_previous.is_stopword), label))
                self.add_binary_feature(
                    'B14=%d_%s' % (int(info.is_punctuation), label)
                )
                self.add_binary_feature(
                    'B15=%d_%s' % (int(info_previous.is_punctuation), label)
                )
                # Ablated for German (capitalized words are nouns)
                # self.add_binary_feature(
                #     'B16=%d_%s' % (int(info.is_proper_noun), label))
                # self.add_binary_feature(
                #     'B17=%d_%s' % (int(info_previous.is_proper_noun), label))
                self.add_binary_feature(
                    'B18=%d_%s' % (int(info.is_digit), label)
                )
                self.add_binary_feature(
                    'B19=%d_%s' % (int(info_previous.is_digit), label)
                )

            if use_language_model and not only_basic_features:
                self.add_binary_feature(
                    'B20=%d_%s' % (info.highest_order_ngram_left, label)
                )
                self.add_binary_feature(
                    'B21=%d_%s' % (info.highest_order_ngram_right, label)
                )
                self.add_binary_feature(
                    'B22=%d_%s'
                    % (info_previous.highest_order_ngram_left, label)
                )
                self.add_binary_feature(
                    'B23=%d_%s'
                    % (info_previous.highest_order_ngram_right, label)
                )

            # if use_language_model and not only_basic_features:
            #     self.add_binary_feature(
            #         'B24=%d_%s' % (info.backoff_behavior_left, label))
            #     self.add_binary_feature(
            #         'B25=%d_%s' % (info.backoff_behavior_middle, label))
            #     self.add_binary_feature(
            #         'B26=%d_%s' % (info.backoff_behavior_right, label))
            #     self.add_binary_feature(
            #         'B27=%d_%s' % (info_previous.backoff_behavior_left,
            #                        label))
            #     self.add_binary_feature(
            #         'B28=%d_%s' % (info_previous.backoff_behavior_middle,
            #                        label))
            #     self.add_binary_feature(
            #         'B29=%d_%s' % (info_previous.backoff_behavior_right,
            #                        label))

            if use_language_model and not only_basic_features:
                self.add_binary_feature(
                    'B30=%d_%s' % (info.source_highest_order_ngram_left, label)
                )
                self.add_binary_feature(
                    'B31=%d_%s' % (info.source_highest_order_ngram_right, label)
                )
                self.add_binary_feature(
                    'B33=%d_%s'
                    % (info_previous.source_highest_order_ngram_left, label)
                )
                self.add_binary_feature(
                    'B34=%d_%s'
                    % (info_previous.source_highest_order_ngram_right, label)
                )

            if not only_basic_features:
                self.add_binary_feature('B35=%s_%s' % (info.target_pos, label))
                self.add_binary_feature(
                    'B36=%s_%s' % (info.aligned_source_pos_list, label)
                )
                self.add_binary_feature(
                    'B37=%s_%s' % (info_previous.target_pos, label)
                )
                self.add_binary_feature(
                    'B38=%s_%s' % (info_previous.aligned_source_pos_list, label)
                )

            # Conjoined features.
            self.add_binary_feature(
                'C0=%s_%s_%s' % (info.token, info.left_context, label)
            )
            self.add_binary_feature(
                'C1=%s_%s_%s' % (info.token, info.right_context, label)
            )
            self.add_binary_feature(
                'C2=%s_%s_%s'
                % (info_previous.token, info_previous.left_context, label)
            )
            self.add_binary_feature(
                'C3=%s_%s_%s'
                % (info_previous.token, info_previous.right_context, label)
            )

            if use_trigram_features:
                self.add_binary_feature(
                    'D1=%s_%s_%s_%s'
                    % (
                        info_previous.left_context,
                        info_previous.token,
                        info.token,
                        label,
                    )
                )
                self.add_binary_feature(
                    'D2=%s_%s_%s_%s'
                    % (
                        info_previous.token,
                        info.token,
                        info.right_context,
                        label,
                    )
                )

            if not ignore_source:
                self.add_binary_feature(
                    'C4=%s_%s_%s'
                    % (info.token, info.first_aligned_token, label)
                )
                self.add_binary_feature(
                    'C5=%s_%s_%s'
                    % (
                        info_previous.token,
                        info_previous.first_aligned_token,
                        label,
                    )
                )
            if not only_basic_features:
                self.add_binary_feature(
                    'C6=%s_%s_%s'
                    % (info.target_pos, info.aligned_source_pos_list, label)
                )
                self.add_binary_feature(
                    'C7=%s_%s_%s'
                    % (
                        info_previous.target_pos,
                        info_previous.aligned_source_pos_list,
                        label,
                    )
                )

            # Parse features.
            if use_parse_features:
                head_index = info.target_head - 1
                previous_head_index = info_previous.target_head - 1
                if head_index >= 0:
                    head_info = sentence_word_features[head_index]
                    head_token = head_info.token
                    head_pos = head_info.target_pos
                    head_morph = head_info.target_morph
                elif head_index == -1:
                    head_info = None
                    head_token = '__ROOT__'
                    head_pos = '__ROOT__'
                    head_morph = '__ROOT__'
                else:
                    head_info = None
                    head_token = '__START__'
                    head_pos = '__START__'
                    head_morph = '__START__'
                if previous_head_index >= 0:
                    previous_head_info = sentence_word_features[
                        previous_head_index
                    ]
                    previous_head_token = previous_head_info.token
                    previous_head_pos = previous_head_info.target_pos
                    previous_head_morph = previous_head_info.target_morph
                elif previous_head_index == -1:
                    previous_head_info = None
                    previous_head_token = '__ROOT__'
                    previous_head_pos = '__ROOT__'
                    previous_head_morph = '__ROOT__'
                else:
                    previous_head_info = None
                    previous_head_token = '__START__'
                    previous_head_pos = '__START__'
                    previous_head_morph = '__START__'

                left_sibling_token, left_sibling_pos, right_sibling_token, right_sibling_pos = self.get_siblings(  # NOQA
                    sentence_word_features, index
                )

                previous_left_sibling_token, previous_left_sibling_pos, previous_right_sibling_token, previous_right_sibling_pos = self.get_siblings(  # NOQA
                    sentence_word_features, index - 1
                )

                if use_deprel_features:
                    self.add_binary_feature(
                        'D0=%s_%s' % (info_previous.target_deprel, label)
                    )
                    self.add_binary_feature(
                        'D1=%s_%s_%s'
                        % (
                            info_previous.token,
                            info_previous.target_deprel,
                            label,
                        )
                    )

                if use_head_features:
                    self.add_binary_feature(
                        'D2=%s_%s_%s'
                        % (info_previous.target_pos, previous_head_pos, label)
                    )
                    self.add_binary_feature(
                        'D3=%s_%s_%s'
                        % (info_previous.token, previous_head_token, label)
                    )

                if use_morph_features:
                    self.add_binary_feature(
                        'D4=%s_%s' % (info_previous.target_morph, label)
                    )
                    self.add_binary_feature(
                        'D5=%s_%s_%s'
                        % (
                            info_previous.target_morph,
                            previous_head_morph,
                            label,
                        )
                    )

                if use_split_morphs:
                    all_morphs = info_previous.target_morph.split('|')
                    all_head_morphs = previous_head_morph.split('|')
                    for m in all_morphs:
                        self.add_binary_feature('D6=%s_%s' % (m, label))
                        for hm in all_head_morphs:
                            self.add_binary_feature(
                                'D7=%s_%s_%s' % (m, hm, label)
                            )

                if use_sibling_features:
                    self.add_binary_feature(
                        'D8=%s_%s_%s'
                        % (
                            info_previous.target_pos,
                            previous_left_sibling_pos,
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'D9=%s_%s_%s'
                        % (
                            info_previous.token,
                            previous_left_sibling_token,
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'D10=%s_%s_%s'
                        % (
                            info_previous.target_pos,
                            previous_right_sibling_pos,
                            label,
                        )
                    )
                    self.add_binary_feature(
                        'D11=%s_%s_%s'
                        % (
                            info_previous.token,
                            previous_right_sibling_token,
                            label,
                        )
                    )

                if use_deprel_features:
                    self.add_binary_feature(
                        'E0=%s_%s' % (info.target_deprel, label)
                    )
                    self.add_binary_feature(
                        'E1=%s_%s_%s' % (info.token, info.target_deprel, label)
                    )

                if use_head_features:
                    self.add_binary_feature(
                        'E2=%s_%s_%s' % (info.target_pos, head_pos, label)
                    )
                    self.add_binary_feature(
                        'E3=%s_%s_%s' % (info.token, head_token, label)
                    )

                if use_morph_features:
                    self.add_binary_feature(
                        'E4=%s_%s' % (info.target_morph, label)
                    )
                    self.add_binary_feature(
                        'E5=%s_%s_%s' % (info.target_morph, head_morph, label)
                    )

                if use_split_morphs:
                    all_morphs = info.target_morph.split('|')
                    all_head_morphs = head_morph.split('|')
                    for m in all_morphs:
                        self.add_binary_feature('E6=%s_%s' % (m, label))
                        for hm in all_head_morphs:
                            self.add_binary_feature(
                                'E7=%s_%s_%s' % (m, hm, label)
                            )

                if use_sibling_features:
                    self.add_binary_feature(
                        'E8=%s_%s_%s'
                        % (info.target_pos, left_sibling_pos, label)
                    )
                    self.add_binary_feature(
                        'E9=%s_%s_%s' % (info.token, left_sibling_token, label)
                    )
                    self.add_binary_feature(
                        'E10=%s_%s_%s'
                        % (info.target_pos, right_sibling_pos, label)
                    )
                    self.add_binary_feature(
                        'E11=%s_%s_%s'
                        % (info.token, right_sibling_token, label)
                    )
            if use_stacked_features:
                if len(info.stacked_features) > 0:
                    for i, value in enumerate(info.stacked_features):
                        self.add_numeric_feature('Z%d_%s' % (i, label), value)
                if len(info_previous.stacked_features) > 0:
                    for i, value in enumerate(info_previous.stacked_features):
                        self.add_numeric_feature('ZZ%d_%s' % (i, label), value)

        if self.save_to_cache:
            self.save_cached_features()
            return
