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


class Corpus:
    def __init__(self, fields_examples=None, dataset_fields=None):
        """Create a Corpus by specifying examples and fields.
        Arguments:
            fields_examples: A list of lists of field values per example.
            dataset_fields: A list of pairs (field name, field object).
        Both lists have the same size (number of fields).
        """
        self.fields_examples = (
            fields_examples if fields_examples is not None else []
        )
        self.dataset_fields = (
            dataset_fields if dataset_fields is not None else []
        )
        self.number_of_examples = (
            len(self.fields_examples[0]) if self.fields_examples else 0
        )

    def examples_per_field(self):
        examples = {
            field: examples
            for (field, _), examples in zip(
                self.dataset_fields, self.fields_examples
            )
        }
        return examples

    @classmethod
    def from_files(cls, fields, files):
        """Create a QualityEstimationDataset given paths and fields.

        Arguments:
            fields: A dict between field name and field object.
            files: A dict between field name and file dict (with 'name' and
                   'format' keys).
        """
        fields_examples = []
        dataset_fields = []

        # first load the data for each field
        for attrib_name, field in fields.items():
            file_dict = files[attrib_name]
            file_name = file_dict['name']
            reader = file_dict['reader']
            if not reader:
                with open(file_name, 'r', encoding='utf8') as f:
                    fields_values_for_example = [line.strip() for line in f]
            else:
                fields_values_for_example = reader(file_name)
            fields_examples.append(fields_values_for_example)
            dataset_fields.append((attrib_name, field))

        # then add each corresponding sentence from each field
        nb_lines = [len(fe) for fe in fields_examples]
        assert min(nb_lines) == max(nb_lines)  # Assert files have the same size
        return cls(fields_examples, dataset_fields)

    @staticmethod
    def read_tabular_file(file_path, sep='\t', extract_column=None):
        examples = []
        line_values = []
        with open(file_path, 'r', encoding='utf8') as f:
            num_columns = None
            for line_num, line in enumerate(f):
                line = line.rstrip()
                if line:
                    values = line.split(sep)
                    line_values.append(values)
                    if num_columns is None:
                        num_columns = len(values)
                        if extract_column is not None and (
                            extract_column < 1 or extract_column > num_columns
                        ):
                            raise IndexError(
                                'Cannot extract column {} (of {})'.format(
                                    extract_column, num_columns
                                )
                            )
                    elif len(values) != num_columns:
                        raise IndexError(
                            'Number of columns ({}) in line {} is different '
                            '({}) for file: {}'.format(
                                len(values),
                                line_num + 1,
                                num_columns,
                                file_path,
                            )
                        )
                else:
                    if extract_column is not None:
                        examples.append(
                            ' '.join(
                                [
                                    values[extract_column - 1]
                                    for values in line_values
                                ]
                            )
                        )
                    else:
                        examples.append(
                            [
                                ' '.join([values[i] for values in line_values])
                                for i in range(num_columns)
                            ]
                        )

                    line_values = []
        if line_values:  # Add trailing lines before EOF.
            if extract_column is not None:
                examples.append(
                    ' '.join(
                        [values[extract_column - 1] for values in line_values]
                    )
                )
            else:
                examples.append(
                    [
                        ' '.join([values[i] for values in line_values])
                        for i in range(num_columns)
                    ]
                )

        return examples

    @classmethod
    def from_tabular_file(cls, fields, file_fields, file_path, sep='\t'):
        """Create a QualityEstimationDataset given paths and fields.

        Arguments:
            fields: A dict between field name and field object.
            file_fields: A list of field names for each column of the file
                (by order). File fields not in fields will be ignored, but
                every field in fields should correspond to some column.
            file_path: Path to the tabular file.
        """
        fields_examples = []
        dataset_fields = []
        examples = {field_name: [] for field_name in fields.keys()}
        example_values = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.rstrip()
                if line:
                    values = line.split(sep)
                    example_values.append(values)
                else:
                    for i, field_name in enumerate(file_fields):
                        if field_name not in fields:  # TODO
                            continue
                        examples[field_name].append(
                            ' '.join([values[i] for values in example_values])
                        )
                    example_values = []
        if example_values:  # Add trailing lines before EOF.
            for i, field_name in enumerate(file_fields):
                if field_name not in fields:
                    continue
                examples[field_name].append(
                    ' '.join([values[i] for values in example_values])
                )

        for attrib_name, field in fields.items():
            fields_examples.append(examples[attrib_name])
            dataset_fields.append((attrib_name, field))

        # then add each corresponding sentence from each field
        nb_lines = [len(fe) for fe in fields_examples]
        assert min(nb_lines) == max(nb_lines)  # Assert files have the same size
        return cls(fields_examples, dataset_fields)

    def __iter__(self):
        for j in range(self.number_of_examples):
            fields_values_for_example = [
                self.fields_examples[i][j]
                for i in range(len(self.dataset_fields))
            ]
            yield data.Example.fromlist(
                fields_values_for_example, self.dataset_fields
            )

    def paste_fields(self, corpus):
        """Pastes (appends) fields from another corpus.
        Arguments:
            corpus: A corpus object. Must have the same number of examples as
                the current corpus.
        """
        assert self.number_of_examples == corpus.number_of_examples
        self.fields_examples += corpus.fields_examples
        self.dataset_fields += corpus.dataset_fields
