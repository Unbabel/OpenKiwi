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

import torch
from torchtext import data

from kiwi import constants as const
from kiwi.data import utils
from kiwi.data.fields.sequence_labels_field import SequenceLabelsField
from kiwi.data.fieldsets.fieldset import Fieldset
from kiwi.data.tokenizers import tokenizer


def build_text_field():
    return data.Field(
        tokenize=tokenizer,
        init_token=const.START,
        batch_first=True,
        eos_token=const.STOP,
        pad_token=const.PAD,
        unk_token=const.UNK,
    )


def build_label_field(postprocessing=None):
    return SequenceLabelsField(
        classes=const.LABELS,
        tokenize=tokenizer,
        pad_token=const.PAD,
        batch_first=True,
        postprocessing=postprocessing,
    )


def build_fieldset(wmt18_format=False):
    target_field = build_text_field()
    source_field = build_text_field()

    source_vocab_options = dict(
        min_freq='source_vocab_min_frequency', max_size='source_vocab_size'
    )
    target_vocab_options = dict(
        min_freq='target_vocab_min_frequency', max_size='target_vocab_size'
    )

    fieldset = Fieldset()
    fieldset.add(
        name=const.SOURCE,
        field=source_field,
        file_option_suffix='_source',
        required=Fieldset.TRAIN,
        vocab_options=source_vocab_options,
    )
    fieldset.add(
        name=const.TARGET,
        field=target_field,
        file_option_suffix='_target',
        required=Fieldset.TRAIN,
        vocab_options=target_vocab_options,
    )
    fieldset.add(
        name=const.PE,
        field=target_field,
        file_option_suffix='_pe',
        required=None,
        vocab_options=target_vocab_options,
    )

    post_pipe_target = data.Pipeline(utils.project)
    if wmt18_format:
        post_pipe_gaps = data.Pipeline(utils.wmt18_to_gaps)
        post_pipe_target = data.Pipeline(utils.wmt18_to_target)
        fieldset.add(
            name=const.GAP_TAGS,
            field=build_label_field(post_pipe_gaps),
            file_option_suffix='_target_tags',
            required=None,
        )

    fieldset.add(
        name=const.TARGET_TAGS,
        field=build_label_field(post_pipe_target),
        file_option_suffix='_target_tags',
        required=None,
    )
    fieldset.add(
        name=const.SOURCE_TAGS,
        field=build_label_field(),
        file_option_suffix='_source_tags',
        required=None,
    )
    fieldset.add(
        name=const.SENTENCE_SCORES,
        field=data.Field(
            sequential=False, use_vocab=False, dtype=torch.float32
        ),
        file_option_suffix='_sentence_scores',
        required=None,
    )

    pipe = data.Pipeline(utils.hter_to_binary)
    fieldset.add(
        name=const.BINARY,
        field=data.Field(
            sequential=False,
            use_vocab=False,
            dtype=torch.long,
            preprocessing=pipe,
        ),
        file_option_suffix='_sentence_scores',
        required=None,
    )
    return fieldset
