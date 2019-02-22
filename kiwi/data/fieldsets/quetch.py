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

from torchtext import data

from kiwi import constants as const
from kiwi.data import utils
from kiwi.data.fields.alignment_field import AlignmentField
from kiwi.data.fields.qe_field import QEField
from kiwi.data.fields.sequence_labels_field import SequenceLabelsField
from kiwi.data.fieldsets.fieldset import Fieldset
from kiwi.data.tokenizers import align_tokenizer, tokenizer


def build_fieldset(wmt18_format=False):
    fs = Fieldset()

    fs.add(
        name=const.SOURCE,
        field=QEField(
            tokenize=tokenizer,
            init_token=None,
            eos_token=None,
            include_lengths=True,
        ),
        file_option_suffix='_source',
        required=Fieldset.ALL,
        vocab_options=dict(
            min_freq='source_vocab_min_frequency',
            max_size='source_vocab_size',
            rare_with_vectors='keep_rare_words_with_embeddings',
            add_vectors_vocab='add_embeddings_vocab',
        ),
        vocab_vectors='source_embeddings',
    )
    fs.add(
        name=const.TARGET,
        field=QEField(
            tokenize=tokenizer,
            init_token=None,
            eos_token=None,
            include_lengths=True,
        ),
        file_option_suffix='_target',
        required=Fieldset.ALL,
        vocab_options=dict(
            min_freq='target_vocab_min_frequency',
            max_size='target_vocab_size',
            rare_with_vectors='keep_rare_words_with_embeddings',
            add_vectors_vocab='add_embeddings_vocab',
        ),
        vocab_vectors='target_embeddings',
    )

    fs.add(
        name=const.ALIGNMENTS,
        field=AlignmentField(tokenize=align_tokenizer, use_vocab=False),
        file_option_suffix='_alignments',
        required=Fieldset.ALL,
    )

    post_pipe_target = data.Pipeline(utils.project)
    if wmt18_format:
        post_pipe_gaps = data.Pipeline(utils.wmt18_to_gaps)
        post_pipe_target = data.Pipeline(utils.wmt18_to_target)
        fs.add(
            name=const.GAP_TAGS,
            field=SequenceLabelsField(
                classes=const.LABELS,
                tokenize=tokenizer,
                pad_token=const.PAD,
                unk_token=None,
                batch_first=True,
                # eos_token=const.STOP,
                postprocessing=post_pipe_gaps,
            ),
            file_option_suffix='_target_tags',
            required=[Fieldset.TRAIN, Fieldset.VALID],
        )

    fs.add(
        name=const.TARGET_TAGS,
        field=SequenceLabelsField(
            classes=const.LABELS,
            tokenize=tokenizer,
            pad_token=const.PAD,
            unk_token=None,
            batch_first=True,
            postprocessing=post_pipe_target,
        ),
        file_option_suffix='_target_tags',
        required=[Fieldset.TRAIN, Fieldset.VALID],
    )
    fs.add(
        name=const.SOURCE_TAGS,
        field=SequenceLabelsField(
            classes=const.LABELS,
            tokenize=tokenizer,
            pad_token=const.PAD,
            unk_token=None,
            batch_first=True,
        ),
        file_option_suffix='_source_tags',
        required=None,
    )

    return fs
