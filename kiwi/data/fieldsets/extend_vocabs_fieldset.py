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
