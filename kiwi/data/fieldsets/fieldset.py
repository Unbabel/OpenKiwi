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

from functools import partial

from kiwi.data.vectors import AvailableVectors


class Fieldset:
    ALL = 'all'
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

    def __init__(self):
        """

        """
        self._fields = {}
        self._options = {}
        self._required = {}
        self._vocab_options = {}
        self._vocab_vectors = {}
        self._file_reader = {}

    def add(
        self,
        name,
        field,
        file_option_suffix,
        required=ALL,
        vocab_options=None,
        vocab_vectors=None,
        file_reader=None,
    ):
        """

        Args:
            name:
            field:
            file_option_suffix:
            required (str or list or None):
            file_reader (callable): by default, uses Corpus.from_files().

        Returns:

        """
        self._fields[name] = field
        self._options[name] = file_option_suffix
        if not isinstance(required, list):
            required = [required]
        self._required[name] = required
        self._file_reader[name] = file_reader

        if vocab_options is None:
            vocab_options = {}
        self._vocab_options[name] = vocab_options
        self._vocab_vectors[name] = vocab_vectors

    @property
    def fields(self):
        return self._fields

    def is_required(self, name, set_name):
        required = self._required[name]
        if set_name in required or self.ALL in required:
            return True
        else:
            return False

    def fields_and_files(self, set_name, **files_options):
        fields = {}
        files = {}
        for name, file_option_suffix in self._options.items():
            file_option = '{}{}'.format(set_name, file_option_suffix)
            file_name = files_options.get(file_option)
            if not file_name and self.is_required(name, set_name):
                raise FileNotFoundError(
                    'File {} is required (use the {} '
                    'option).'.format(file_name, file_option.replace('_', '-'))
                )
            elif file_name:
                files[name] = {
                    'name': file_name,
                    'reader': self._file_reader.get(name),
                }
                fields[name] = self._fields[name]
        return fields, files

    # def files_formats(self):
    #     return {
    #         set_name: self._file_format.get(set_name)
    #         for set_name in self._fields
    #     }
    #
    def vocab_kwargs(self, name, **kwargs):
        if name not in self._vocab_options:
            raise KeyError(
                'Field named "{}" does not exist in this fieldset'.format(name)
            )
        vkwargs = {}
        for argument, option_name in self._vocab_options[name].items():
            option_value = kwargs.get(option_name)
            if option_value is not None:
                vkwargs[argument] = option_value
        return vkwargs

    def vocab_vectors_loader(
        self,
        name,
        embeddings_format='polyglot',
        embeddings_binary=False,
        **kwargs
    ):
        if name not in self._vocab_vectors:
            raise KeyError(
                'Field named "{}" does not exist in this fieldset'.format(name)
            )

        def no_vectors_fn():
            return None

        vectors_fn = no_vectors_fn

        option_name = self._vocab_vectors[name]
        if option_name:
            option_value = kwargs.get(option_name)
            if option_value:
                emb_model = AvailableVectors[embeddings_format]
                # logger.info('Loading {} embeddings from {}'.format(
                #     name, option_value))
                vectors_fn = partial(
                    emb_model, option_value, binary=embeddings_binary
                )
        return vectors_fn

    def vocab_vectors(self, name, **kwargs):
        vectors_fn = self.vocab_vectors_loader(name, **kwargs)
        return vectors_fn()

    def fields_vocab_options(self, **kwargs):
        vocab_options = {}
        for name, field in self.fields.items():
            vocab_options[name] = dict(
                vectors_fn=self.vocab_vectors_loader(name, **kwargs)
            )
            vocab_options[name].update(self.vocab_kwargs(name, **kwargs))
        return vocab_options
