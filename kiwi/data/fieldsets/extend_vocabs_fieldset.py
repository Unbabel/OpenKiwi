from kiwi import constants as const
from kiwi.data.fieldsets.fieldset import Fieldset


def build_fieldset(base_fieldset):
    source_field = base_fieldset.fields[const.SOURCE]
    target_field = base_fieldset.fields[const.TARGET]

    source_vocab_options = dict(
        min_freq='source_vocab_min_frequency', max_size='source_vocab_size'
    )
    target_vocab_options = dict(
        min_freq='target_vocab_min_frequency', max_size='target_vocab_size'
    )

    extend_vocabs = Fieldset()
    extend_vocabs.add(
        name=const.SOURCE,
        field=source_field,
        file_option_suffix='extend_source_vocab',
        required=None,
        vocab_options=source_vocab_options,
    )
    extend_vocabs.add(
        name=const.TARGET,
        field=target_field,
        file_option_suffix='extend_target_vocab',
        required=None,
        vocab_options=target_vocab_options,
    )

    return extend_vocabs
