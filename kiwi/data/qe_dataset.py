from torchtext import data


class QEDataset(data.Dataset):
    """Defines a dataset for quality estimation. Based on the WMT 201X."""

    @staticmethod
    def sort_key(ex):
        # don't work for pack_padded_sequences
        # return data.interleave_keys(len(ex.source), len(ex.target))
        return len(ex.source)

    def __init__(self, examples, fields, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.

        Arguments:
            examples: List of Examples.
            fields (List(tuple(str, Field))): The Fields to use in this tuple.
                The string is a field name, and the Field is the associated
                field.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        # ensure that examples is not a generator
        examples = list(examples)
        super().__init__(examples, fields, filter_pred)

    def __getstate__(self):
        """For pickling. Copied from OpenNMT-py DatasetBase implementation.
        """
        return self.__dict__

    def __setstate__(self, _d):
        """For pickling. Copied from OpenNMT-py DatasetBase implementation.
        """
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        """For pickling. Copied from OpenNMT-py DatasetBase implementation.
        """
        return super(QEDataset, self).__reduce_ex__(proto)

    def split(
        self,
        split_ratio=0.7,
        stratified=False,
        strata_field='label',
        random_state=None,
    ):
        datasets = super().split(
            split_ratio, stratified, strata_field, random_state
        )
        casted_datasets = [
            QEDataset(examples=dataset.examples, fields=dataset.fields)
            for dataset in datasets
        ]
        return casted_datasets
