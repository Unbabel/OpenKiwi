from torchtext import data

from kiwi import constants
from kiwi.data.fields.sequence_labels_field import SequenceLabelsField
from kiwi.data.fieldsets.fieldset import Fieldset
from kiwi.data.tokenizers import tokenizer


def build_text_field():
    return data.Field(
        tokenize=tokenizer,
        init_token=constants.START,
        batch_first=True,
        eos_token=constants.STOP,
        pad_token=constants.PAD,
        unk_token=constants.UNK,
    )


def build_label_field(postprocessing=None):
    return SequenceLabelsField(
        classes=constants.LABELS,
        tokenize=tokenizer,
        pad_token=constants.PAD,
        batch_first=True,
        postprocessing=postprocessing,
    )


def build_fieldset():
    source_field = build_text_field()
    target_field = build_text_field()

    source_vocab_options = dict(
        min_freq='source_vocab_min_frequency', max_size='source_vocab_size'
    )
    target_vocab_options = dict(
        min_freq='target_vocab_min_frequency', max_size='target_vocab_size'
    )

    fieldset = Fieldset()
    fieldset.add(
        name=constants.SOURCE,
        field=source_field,
        file_option_suffix='_source',
        required=Fieldset.TRAIN,
        vocab_options=source_vocab_options,
    )
    fieldset.add(
        name=constants.TARGET,
        field=target_field,
        file_option_suffix='_target',
        required=Fieldset.TRAIN,
        vocab_options=target_vocab_options,
    )
    return fieldset
