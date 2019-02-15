from kiwi import constants as const
from kiwi.data.fields.qe_field import QEField
from kiwi.data.fieldsets.fieldset import Fieldset
from kiwi.data.tokenizers import tokenizer


def build_text_field():
    return QEField(
        tokenize=tokenizer,
        init_token=const.START,
        batch_first=True,
        eos_token=const.STOP,
        pad_token=const.PAD,
        unk_token=const.UNK,
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
    return fieldset
