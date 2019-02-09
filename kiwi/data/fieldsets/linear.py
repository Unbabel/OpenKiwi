from functools import partial

from torchtext import data

from kiwi import constants
from kiwi.data.corpus import Corpus
from kiwi.data.fields.alignment_field import AlignmentField
from kiwi.data.fields.qe_field import QEField
from kiwi.data.fieldsets.fieldset import Fieldset
from kiwi.data.tokenizers import align_tokenizer, tokenizer


def build_fieldset():
    fs = Fieldset()

    source_vocab_options = dict(
        min_freq='source_vocab_min_frequency', max_size='source_vocab_size'
    )
    target_vocab_options = dict(
        min_freq='target_vocab_min_frequency', max_size='target_vocab_size'
    )

    source_field = QEField(tokenize=tokenizer)
    target_field = QEField(tokenize=tokenizer)
    source_pos = QEField(tokenize=tokenizer)
    target_pos = QEField(tokenize=tokenizer)
    target_tags_field = data.Field(
        tokenize=tokenizer, pad_token=None, unk_token=None
    )

    fs.add(
        name=constants.SOURCE,
        field=source_field,
        file_option_suffix='_source',
        required=Fieldset.ALL,
        vocab_options=source_vocab_options,
    )
    fs.add(
        name=constants.TARGET,
        field=target_field,
        file_option_suffix='_target',
        required=Fieldset.ALL,
        vocab_options=target_vocab_options,
    )
    fs.add(
        name=constants.ALIGNMENTS,
        field=AlignmentField(tokenize=align_tokenizer, use_vocab=False),
        file_option_suffix='_alignments',
        required=Fieldset.ALL,
    )
    fs.add(
        name=constants.TARGET_TAGS,
        field=target_tags_field,
        file_option_suffix='_target_tags',
        required=[Fieldset.TRAIN, Fieldset.VALID],
    )

    fs.add(
        name=constants.SOURCE_POS,
        field=source_pos,
        file_option_suffix='_source_pos',
        required=None,
    )
    fs.add(
        name=constants.TARGET_POS,
        field=target_pos,
        file_option_suffix='_target_pos',
        required=None,
    )

    target_stacked = data.Field(tokenize=tokenizer)
    fs.add(
        name=constants.TARGET_STACKED,
        field=target_stacked,
        file_option_suffix='_target_stacked',
        file_reader=partial(Corpus.read_tabular_file, extract_column=1),
        required=None,
    )

    target_parse_heads = data.Field(tokenize=tokenizer, use_vocab=False)
    target_parse_relations = data.Field(tokenize=tokenizer)
    fs.add(
        name=constants.TARGET_PARSE_HEADS,
        field=target_parse_heads,
        file_option_suffix='_target_parse',
        file_reader=partial(Corpus.read_tabular_file, extract_column=1),
        required=None,
    )
    fs.add(
        name=constants.TARGET_PARSE_RELATIONS,
        field=target_parse_relations,
        file_option_suffix='_target_parse',
        file_reader=partial(Corpus.read_tabular_file, extract_column=2),
        required=None,
    )

    target_ngram_left = data.Field(tokenize=tokenizer)
    target_ngram_right = data.Field(tokenize=tokenizer)
    fs.add(
        name=constants.TARGET_NGRAM_LEFT,
        field=target_ngram_left,
        file_option_suffix='_target_ngram',
        file_reader=partial(Corpus.read_tabular_file, extract_column=1),
        required=None,
    )
    fs.add(
        name=constants.TARGET_NGRAM_RIGHT,
        field=target_ngram_right,
        file_option_suffix='_target_ngram',
        file_reader=partial(Corpus.read_tabular_file, extract_column=2),
        required=None,
    )

    return fs


#
# def build_test_dataset(options):
#     source_field = QEField(tokenize=tokenizer)
#     target_field = QEField(tokenize=tokenizer)
#     source_pos = QEField(tokenize=tokenizer)
#     target_pos = QEField(tokenize=tokenizer)
#     alignments_field = AlignmentField(
#         tokenize=align_tokenizer, use_vocab=False)
#     target_tags_field = data.Field(
#         tokenize=tokenizer, pad_token=None, unk_token=None
#     )
#     target_parse_heads = data.Field(tokenize=tokenizer, use_vocab=False)
#     target_parse_relations = data.Field(tokenize=tokenizer)
#     target_ngram_left = data.Field(tokenize=tokenizer)
#     target_ngram_right = data.Field(tokenize=tokenizer)
#     target_stacked = data.Field(tokenize=tokenizer)
#
#     fields = {
#         constants.SOURCE: source_field,
#         constants.TARGET: target_field,
#         constants.ALIGNMENTS: alignments_field,
#         constants.TARGET_TAGS: target_tags_field
#     }
#
#     test_files = {
#         constants.SOURCE: options.test_source,
#         constants.TARGET: options.test_target,
#         constants.TARGET_TAGS: options.test_target_tags,
#         constants.ALIGNMENTS: options.test_alignments,
#     }
#
#     if options.test_target_parse:
#         parse_fields = {
#             constants.TARGET_PARSE_HEADS: target_parse_heads,
#             constants.TARGET_PARSE_RELATIONS: target_parse_relations,
#         }
#         parse_file_fields = [
#             '',
#             '',
#             '',
#             '',
#             '',
#             constants.TARGET_PARSE_HEADS,
#             constants.TARGET_PARSE_RELATIONS,
#         ]
#
#     if options.test_target_ngram:
#         ngram_fields = {
#             constants.TARGET_NGRAM_LEFT: target_ngram_left,
#             constants.TARGET_NGRAM_RIGHT: target_ngram_right,
#         }
#         ngram_file_fields = [
#             '', '', '', '', '', '', '', '', '', '', '', '', '',
#             constants.TARGET_NGRAM_LEFT,
#             constants.TARGET_NGRAM_RIGHT,
#         ]
#
#     if options.test_target_stacked:
#         stacked_fields = {constants.TARGET_STACKED: target_stacked}
#         stacked_file_fields = [constants.TARGET_STACKED]
#
#     if options.test_source_pos:
#         fields[constants.SOURCE_POS] = source_pos
#         test_files[constants.SOURCE_POS] = options.test_source_pos
#
#     if options.test_target_pos:
#         fields[constants.TARGET_POS] = target_pos
#         test_files[constants.TARGET_POS] = options.test_target_pos
#
#     if options.test_target_parse:
#         test_target_parse_file = options.test_target_parse
#
#     if options.test_target_ngram:
#         test_target_ngram_file = options.test_target_ngram
#
#     if options.test_target_stacked:
#         test_target_stacked_file = options.test_target_stacked
#
#     def filter_len(x):
#         return (
#                    options.source_min_length
#                    <= len(x.source)
#                    <= options.source_max_length
#                ) and (
#                    options.target_min_length
#                    <= len(x.target)
#                    <= options.target_max_length
#                )
#
#     test_examples = Corpus.from_files(fields=fields, files=test_files)
#     if options.test_target_parse:
#         test_examples.paste_fields(
#             Corpus.from_tabular_file(
#                 fields=parse_fields,
#                 file_fields=parse_file_fields,
#                 file_path=test_target_parse_file,
#             )
#         )
#     if options.test_target_ngram:
#         test_examples.paste_fields(
#             Corpus.from_tabular_file(
#                 fields=ngram_fields,
#                 file_fields=ngram_file_fields,
#                 file_path=test_target_ngram_file,
#             )
#         )
#     if options.test_target_stacked:
#         test_examples.paste_fields(
#             Corpus.from_tabular_file(
#                 fields=stacked_fields,
#                 file_fields=stacked_file_fields,
#                 file_path=test_target_stacked_file,
#             )
#         )
#
#     dataset = QEDataset(
#         examples=test_examples,
#         fields=test_examples.dataset_fields,
#         filter_pred=filter_len,
#     )
#
#     return dataset
