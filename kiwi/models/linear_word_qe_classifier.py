"""This is the main script for the linear sequential word-based quality
estimator."""
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

import logging
from pathlib import Path

import numpy as np

from kiwi import constants as const
from kiwi.data.fieldsets.linear import build_fieldset
from kiwi.models.linear.label_dictionary import LabelDictionary
from kiwi.models.linear.linear_word_qe_decoder import LinearWordQEDecoder
from kiwi.models.linear.linear_word_qe_features import LinearWordQEFeatures
from kiwi.models.linear.linear_word_qe_sentence import LinearWordQESentence
from kiwi.models.linear.sequence_parts import (
    SequenceBigramPart,
    SequenceUnigramPart,
)
from kiwi.models.linear.structured_classifier import StructuredClassifier

logger = logging.getLogger(__name__)


class LinearWordQEClassifier(StructuredClassifier):
    """Main class for the word-level quality estimator. Inherits from a
    general structured classifier."""

    title = 'Linear Model'

    def __init__(
        self,
        use_basic_features_only=True,
        use_bigrams=True,
        use_simple_bigram_features=True,
        use_parse_features=False,
        use_stacked_features=False,
        evaluation_metric='f1_bad',
        cost_false_positives=0.5,
        cost_false_negatives=0.5,
    ):
        super().__init__()

        self.decoder = LinearWordQEDecoder(
            self, cost_false_positives, cost_false_negatives
        )
        self.labels = LabelDictionary()

        self.use_basic_features_only = use_basic_features_only
        self.use_bigrams = use_bigrams
        self.use_simple_bigram_features = use_simple_bigram_features
        self.use_parse_features = use_parse_features
        self.use_stacked_features = use_stacked_features

        # Evaluation.
        self.evaluation_metric = evaluation_metric

    @staticmethod
    def fieldset(*args, **kwargs):
        return build_fieldset()

    @staticmethod
    def from_options(vocabs, opts):
        use_parse_features = True if opts.train_target_parse else False
        use_stacked_features = True if opts.train_target_stacked else False
        model = LinearWordQEClassifier(
            use_basic_features_only=opts.use_basic_features_only,
            use_bigrams=opts.use_bigrams,
            use_simple_bigram_features=opts.use_simple_bigram_features,
            use_parse_features=use_parse_features,
            use_stacked_features=use_stacked_features,
            evaluation_metric=opts.evaluation_metric,
            cost_false_positives=opts.cost_false_positives,
            cost_false_negatives=opts.cost_false_negatives,
        )
        return model

    def num_parameters(self):
        return len(self.__dict__)

    # -- END of new methods --

    # TODO: Eliminate this function.
    def get_coarse_label(self, label):
        """Get the coarse part of a fine-grained label. The coarse label is the
        prefix before the underscore (if any). For example, the coarse part of
        BAD_SUB, BAD_DEL, and BAD is BAD."""
        sep = label.find('_')
        if sep >= 0:
            coarse_label = label[:sep]
        else:
            coarse_label = label
        return coarse_label

    def create_instances(self, dataset):
        instances = []
        num_words = 0
        for example in dataset:
            sentence = LinearWordQESentence()
            labels = None
            if hasattr(example, 'tags'):
                labels = []
                for label in example.tags:
                    if label in self.labels:
                        label_id = self.labels.get_label_id(label)
                    else:
                        label_id = self.labels.add(label)
                    labels.append(label_id)

            sentence.create_from_sentence_pair(
                source_words=example.source,
                target_words=example.target,
                alignments=example.alignments,
                source_pos_tags=getattr(example, const.SOURCE_POS, None),
                target_pos_tags=getattr(example, const.TARGET_POS, None),
                target_parse_heads=getattr(
                    example, const.TARGET_PARSE_HEADS, None
                ),
                target_parse_relations=getattr(
                    example, const.TARGET_PARSE_RELATIONS, None
                ),
                target_ngram_left=getattr(
                    example, const.TARGET_NGRAM_LEFT, None
                ),
                target_ngram_right=getattr(
                    example, const.TARGET_NGRAM_RIGHT, None
                ),
                target_stacked_features=getattr(
                    example, const.TARGET_STACKED, None
                ),
                labels=labels,
            )
            instances.append(sentence)
            num_words += sentence.num_words()

        logger.info('Number of sentences: %d' % len(instances))
        logger.info('Number of words: %d' % num_words)
        logger.info('Number of labels: %d' % len(self.labels))

        return instances

    def make_parts(self, instance):
        """Creates the parts (unigrams and bigrams) for an instance."""
        gold_list = []
        parts = []
        make_gold = True
        for word_index in range(instance.num_words()):
            for label_id in range(len(self.labels)):
                part = SequenceUnigramPart(word_index, label_id)
                parts.append(part)
                if make_gold:
                    if label_id == instance.sentence_word_labels[word_index]:
                        gold_list.append(1.0)
                    else:
                        gold_list.append(0.0)
        if self.use_bigrams:
            # First word.
            for label_id in range(len(self.labels)):
                part = SequenceBigramPart(0, label_id, -1)
                parts.append(part)
                if make_gold:
                    if label_id == instance.sentence_word_labels[0]:
                        gold_list.append(1.0)
                    else:
                        gold_list.append(0.0)
            # Intermediate word.
            for word_index in range(1, instance.num_words()):
                for label_id in range(len(self.labels)):
                    for previous_label_id in range(len(self.labels)):
                        part = SequenceBigramPart(
                            word_index, label_id, previous_label_id
                        )
                        parts.append(part)
                        if make_gold:
                            if (
                                label_id
                                == instance.sentence_word_labels[word_index]
                                and previous_label_id
                                == instance.sentence_word_labels[word_index - 1]
                            ):
                                gold_list.append(1.0)
                            else:
                                gold_list.append(0.0)
            # Last word.
            for previous_label_id in range(len(self.labels)):
                part = SequenceBigramPart(
                    instance.num_words(), -1, previous_label_id
                )
                parts.append(part)
                if make_gold:
                    if (
                        previous_label_id
                        == instance.sentence_word_labels[
                            instance.num_words() - 1
                        ]
                    ):
                        gold_list.append(1.0)
                    else:
                        gold_list.append(0.0)

        if make_gold:
            gold_array = np.array(gold_list)
            return parts, gold_array
        else:
            return parts

    def make_features(self, instance, parts):
        """Creates a feature vector for each part."""
        features = []
        for part in parts:
            part_features = LinearWordQEFeatures(
                use_basic_features_only=self.use_basic_features_only,
                use_simple_bigram_features=self.use_simple_bigram_features,
                use_parse_features=self.use_parse_features,
                use_stacked_features=self.use_stacked_features,
            )
            if isinstance(part, SequenceUnigramPart):
                part_features.compute_unigram_features(
                    instance.sentence_word_features, part
                )
            elif isinstance(part, SequenceBigramPart):
                part_features.compute_bigram_features(
                    instance.sentence_word_features, part
                )
            else:
                raise NotImplementedError
            features.append(part_features)
        return features

    def label_instance(self, instance, parts, predicted_output):
        """Return a labeled instance by adding the predicted output
        information."""
        assert False, 'This does not seem to be called'
        labeled_instance = LinearWordQESentence(instance.sentence)
        labeled_instance.sentence_word_features = (
            instance.sentence_word_features
        )
        predictions = np.zeros(instance.num_words(), dtype=int)
        for r, part in enumerate(parts):
            if isinstance(part, SequenceUnigramPart):
                continue
            if predicted_output[r] > 0.5:
                predictions[part.index] = part.label
        labeled_instance.sentence_word_labels = [
            self.labels.get_label_name(pred) for pred in predictions
        ]
        return labeled_instance

    def create_prediction(self, instance, parts, predicted_output):
        """Creates a list of word-level predictions for a sentence.
        For compliance with probabilities, it returns 1 if label is BAD, 0 if
        OK."""
        predictions = np.zeros(instance.num_words(), dtype=int)
        for r, part in enumerate(parts):
            if not isinstance(part, SequenceUnigramPart):
                continue
            if predicted_output[r] > 0.5:
                predictions[part.index] = part.label
        predictions = [
            int(const.BAD == self.labels.get_label_name(pred))
            for pred in predictions
        ]
        return predictions

    def test(self, instances):
        """Run the model on test data."""
        logger.info('Testing...')
        predictions = StructuredClassifier.test(self, instances)
        return predictions

    def evaluate(self, instances, predictions, print_scores=True):
        """Evaluates the model's accuracy and F1-BAD score."""
        all_predictions = []
        for word_predictions in predictions:
            labels = [
                const.BAD if prediction else const.OK
                for prediction in word_predictions
            ]
            labels = [int(self.labels[label]) for label in labels]
            all_predictions.append(labels)

        # TODO: Get rid of fine-grained labels.
        # Allow fine-grained labels. Their names should be a coarse-grained
        # label, followed by an underscore, followed by a sub-label.
        # For example, BAD_SUB or BAD_DEL are two instances of bad labels.
        fine_to_coarse = -np.ones(len(self.labels), dtype=int)
        coarse_labels = LabelDictionary()
        for label in self.labels:
            coarse_label = self.get_coarse_label(label)
            if coarse_label not in coarse_labels:
                lid = coarse_labels.add(coarse_label)
            else:
                lid = coarse_labels[coarse_label]
            fine_to_coarse[self.labels[label]] = lid

        # Iterate through sentences and compare gold values with predicted
        # values. Update counts.
        num_matched = 0
        num_matched_labels = np.zeros(len(coarse_labels))
        num_predicted = 0
        num_predicted_labels = np.zeros(len(coarse_labels))
        num_gold_labels = np.zeros(len(coarse_labels))
        assert len(all_predictions) == len(instances)
        for i, instance in enumerate(instances):
            predictions = all_predictions[i]
            assert len(instance.sentence_word_labels) == len(predictions)
            for j in range(len(predictions)):
                if (
                    fine_to_coarse[predictions[j]]
                    == fine_to_coarse[instance.sentence_word_labels[j]]
                ):
                    num_matched += 1
                num_predicted += 1
                if (
                    fine_to_coarse[predictions[j]]
                    == fine_to_coarse[instance.sentence_word_labels[j]]
                ):
                    num_matched_labels[fine_to_coarse[predictions[j]]] += 1
                num_predicted_labels[fine_to_coarse[predictions[j]]] += 1
                num_gold_labels[
                    fine_to_coarse[instance.sentence_word_labels[j]]
                ] += 1

        acc = float(num_matched) / float(num_predicted)
        logger.info('Accuracy: %f' % acc)

        # We allow multiple bad labels. They should be named BAD*.
        bad = coarse_labels['BAD']
        if num_matched_labels[bad] == 0:
            f1_bad = 0.0
        else:
            precision_bad = float(num_matched_labels[bad]) / float(
                num_predicted_labels[bad]
            )
            recall_bad = float(num_matched_labels[bad]) / float(
                num_gold_labels[bad]
            )
            f1_bad = (
                2 * precision_bad * recall_bad / (precision_bad + recall_bad)
            )

        logger.info(
            '# gold bad: %d/%d' % (num_gold_labels[bad], sum(num_gold_labels))
        )
        logger.info(
            '# predicted bad: %d/%d'
            % (num_predicted_labels[bad], sum(num_predicted_labels))
        )

        ok = coarse_labels['OK']
        if num_matched_labels[ok] == 0:
            f1_ok = 0.0
        else:
            precision_ok = float(num_matched_labels[ok]) / float(
                num_predicted_labels[ok]
            )
            recall_ok = float(num_matched_labels[ok]) / float(
                num_gold_labels[ok]
            )
            f1_ok = 2 * precision_ok * recall_ok / (precision_ok + recall_ok)

        logger.info(
            '# gold ok: %d/%d' % (num_gold_labels[ok], sum(num_gold_labels))
        )
        logger.info(
            '# predicted ok: %d/%d'
            % (num_predicted_labels[ok], sum(num_predicted_labels))
        )

        logger.info('F1 bad: %f' % f1_bad)
        logger.info('F1 ok: %f' % f1_ok)
        logger.info('F1 mult: %f' % (f1_bad * f1_ok))
        if self.evaluation_metric == 'f1_mult':
            return f1_bad * f1_ok
        elif self.evaluation_metric == 'f1_bad':
            return f1_bad
        else:
            raise NotImplementedError

    def load_configuration(self, config):
        self.use_basic_features_only = config['use_basic_features_only']
        self.use_bigrams = config['use_bigrams']
        self.use_simple_bigram_features = config['use_simple_bigram_features']
        self.use_stacked_features = config['use_stacked']
        self.use_parse_features = config['use_parse']

    def save_configuration(self):
        config = {
            'use_basic_features_only': self.use_basic_features_only,
            'use_bigrams': self.use_bigrams,
            'use_simple_bigram_features': self.use_simple_bigram_features,
            'use_stacked': self.use_stacked_features,
            'use_parse': self.use_parse_features,
        }
        return config

    def load(self, model_path):
        import pickle

        with Path(model_path).open('rb') as fid:
            config = pickle.load(fid)
            self.load_configuration(config)
            self.labels = pickle.load(fid)
            self.model = pickle.load(fid)
            try:
                self.source_vocab = pickle.load(fid)
                self.target_vocab = pickle.load(fid)
            except EOFError:
                self.source_vocab = None
                self.target_vocab = None

    def save(self, model_path):
        import pickle

        with Path(model_path).open('wb') as fid:
            config = self.save_configuration()
            pickle.dump(config, fid)
            pickle.dump(self.labels, fid)
            pickle.dump(self.model, fid)
            # pickle.dump(self.source_vocab, fid)
            # pickle.dump(self.target_vocab, fid)
