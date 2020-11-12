#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2020 Unbabel <openkiwi@unbabel.com>
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
import logging
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Union

import torch
from more_itertools import one
from pydantic import validator
from torch import Tensor, nn
from transformers import (
    XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
    AutoTokenizer,
    XLMConfig,
    XLMModel,
)
from transformers.tokenization_xlm import lowercase_and_remove_accent

from kiwi import constants as const
from kiwi.data.encoders.field_encoders import TextEncoder
from kiwi.data.vocabulary import Vocabulary
from kiwi.systems._meta_module import MetaModule
from kiwi.utils.data_structures import DefaultFrozenDict
from kiwi.utils.io import BaseConfig
from kiwi.utils.tensors import pieces_to_tokens, retrieve_tokens_mask, select_positions

logger = logging.getLogger(__name__)


class XLMTextEncoder(TextEncoder):
    def __init__(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name))
        self.xlm_tokenizer = tokenizer

        def subtokenize(token):
            """Split a token into pieces.

            This is essentially the same as ``xlm_tokenizer._tokenize()`` but without
            tokenizing a sentence (according to given language), because QE data is
            already tokenized.
            """
            cleaned_token = lowercase_and_remove_accent([token])
            cleaned_token = one(cleaned_token)  # Must have a single element

            if cleaned_token:
                split_token = self.xlm_tokenizer.bpe(token).split(' ')
            else:
                logger.warning(
                    f'XLM tokenization for token "{token}" returned "{cleaned_token}"; '
                    f'replacing with "."'
                )
                split_token = ['.']

            return split_token

        super().__init__(
            subtokenize=subtokenize,
            pad_token=tokenizer.pad_token,
            unk_token=tokenizer.unk_token,
            bos_token=tokenizer.cls_token,  # we don't use xlm.bos_token on purpose
            eos_token=tokenizer.sep_token,
            specials_first=True,
            # extra options from fields?
            include_lengths=True,
            include_bounds=True,
        )

        self.vocab = Vocabulary(
            counter=Counter(),
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            specials_first=self.specials_first,
        )
        self.vocab.stoi = DefaultFrozenDict(
            tokenizer.encoder, tokenizer.encoder[self.unk_token]
        )
        self.vocab.itos = tokenizer.decoder

    def fit_vocab(
        self,
        samples,
        vocab_size=None,
        vocab_min_freq=0,
        embeddings_name=None,
        keep_rare_words_with_embeddings=False,
        add_embeddings_vocab=False,
    ):
        logger.info(
            'Vocabulary already defined for XLM field; not fitting it to data now'
        )
        if vocab_size:
            self.vocab.max_size(vocab_size)


@MetaModule.register_subclass
class XLMEncoder(MetaModule):
    """XLM model using Hugging Face's transformers library.

    The following command was used to fine-tune XLM on the in-domain data (retrieved
    from .pth file)::

        python train.py --exp_name tlm_clm --dump_path './dumped/' \
            --data_path '/mnt/shared/datasets/kiwi/parallel/en_de_indomain' \
            --lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh' \
            --clm_steps 'en-de,de-en' --mlm_steps 'en-de,de-en' \
            --reload_model 'models/mlm_tlm_xnli15_1024.pth' --encoder_only True \
            --emb_dim 1024 --n_layers 12 --n_heads 8 --dropout '0.1' \
            --attention_dropout '0.1' --gelu_activation true --batch_size 32 \
            --bptt 256 --optimizer
            'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,weight_decay=0' \
            --epoch_size 200000 --validation_metrics _valid_mlm_ppl --max_vocab 95000 \
            --tokens_per_batch 1200 --exp_id "5114"

    Old version was converted using hf-transformers util method::

        convert_xlm_checkpoint_to_pytorch(
            self.config.model_name / 'indomain.pth',
            self.config.model_name / 'finetuned_wmt_en-de'
        )

    Old settings in QE not really used for the best run and submission:

    .. code-block:: yaml

        fb-causal-lambda: 0.0
        fb-keep-prob: 0.1
        fb-mask-prob: 0.8
        fb-model: data/trained_models/fb_pretrain/xnli/indomain.pth
        fb-pred-prob: 0.15
        fb-rand-prob: 0.1
        fb-src-lang: en
        fb-tgt-lang: de
        fb-tlm-lambda: 0.0
        fb-vocab: data/trained_models/fb_pretrain/xnli/vocab_xnli_15.txt

    """

    class Config(BaseConfig):
        model_name: Union[str, Path] = 'xlm-mlm-tlm-xnli15-1024'
        """Pre-trained XLM model to use."""

        source_language: str = 'en'
        target_language: str = 'de'

        use_mismatch_features: bool = False
        """Use Alibaba's mismatch features."""

        use_predictor_features: bool = False
        """Use features originally proposed in the Predictor model."""

        interleave_input: bool = False
        """Concatenate SOURCE and TARGET without internal padding
        (111222000 instead of 111002220)"""

        freeze: bool = False
        """Freeze XLM during training."""

        use_mlp: bool = True
        """Apply a linear layer on top of XLM."""

        hidden_size: int = 100
        """Size of the linear layer on top of XLM."""

        @validator('model_name', pre=True)
        def fix_relative_path(cls, v):
            if v not in XLM_PRETRAINED_MODEL_ARCHIVE_LIST:
                v = Path(v)
                if not v.is_absolute():
                    v = Path.cwd().joinpath(v)
            return v

        @validator('use_mismatch_features', 'use_predictor_features', pre=True)
        def no_implementation(cls, v):
            if v:
                raise NotImplementedError('Not yet implemented')
            return False

    def __init__(
        self, vocabs: Dict[str, Vocabulary], config: Config, pre_load_model: bool = True
    ):
        super().__init__(config=config)

        if pre_load_model:
            self.xlm = XLMModel.from_pretrained(
                self.config.model_name, output_hidden_states=True
            )
        else:
            xlm_config = XLMConfig.from_pretrained(
                self.config.model_name, output_hidden_states=True
            )
            self.xlm = XLMModel(xlm_config)

        self.source_lang_id = self.xlm.config.lang2id.get(self.config.source_language)
        self.target_lang_id = self.xlm.config.lang2id.get(self.config.target_language)

        if None in (self.source_lang_id, self.target_lang_id):
            raise ValueError(
                f'Invalid lang_id for XLM model.'
                f' Valid ids are: {self.xlm.config.lang2id.keys()}'
            )

        self.mlp = None
        if self.config.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.xlm.config.hidden_size, self.config.hidden_size),
                nn.Tanh(),
            )
            output_size = self.config.hidden_size
        else:
            output_size = self.xlm.config.hidden_size

        self._sizes = {
            const.TARGET: output_size,
            const.TARGET_LOGITS: output_size,
            const.TARGET_SENTENCE: 2 * output_size,
            const.SOURCE: output_size,
            const.SOURCE_LOGITS: output_size,
        }

        self.vocabs = {
            const.TARGET: vocabs[const.TARGET],
            const.SOURCE: vocabs[const.SOURCE],
        }

        self.output_embeddings = self.xlm.embeddings

        if self.config.freeze:
            for param in self.xlm.parameters():
                param.requires_grad = False

    def load_state_dict(
        self,
        state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
        strict: bool = True,
    ):
        try:
            keys = super().load_state_dict(state_dict, strict)
        except RuntimeError as e:
            if "position_ids" in str(e):
                # FIXME: hack to get around Transformers 3.1 breaking changes
                # https://github.com/huggingface/transformers/issues/6882
                self.xlm.embeddings._non_persistent_buffers_set.add('position_ids')
                keys = super().load_state_dict(state_dict, strict)
                self.xlm.embeddings._non_persistent_buffers_set.discard('position_ids')
            else:
                raise e
        return keys

    @classmethod
    def input_data_encoders(cls, config: Config):
        return {
            const.SOURCE: XLMTextEncoder(tokenizer_name=config.model_name),
            const.TARGET: XLMTextEncoder(tokenizer_name=config.model_name),
        }

    def size(self, field=None):
        if field:
            return self._sizes[field]
        return self._sizes

    def forward(
        self,
        batch_inputs,
        *args,
        include_target_logits=False,
        include_source_logits=False,
    ):
        # XLM gets it's input as a concatenation of both embeddings
        # or as an interleave of inputs
        if self.config.interleave_input:
            merge_input_fn = self.interleave_input
        else:
            merge_input_fn = self.concat_input

        input_ids, _, attention_mask, position_ids, lang_ids = merge_input_fn(
            batch_a=batch_inputs[const.SOURCE],
            batch_b=batch_inputs[const.TARGET],
            pad_id=self.vocabs[const.TARGET].pad_id,
            lang_a=self.source_lang_id,
            lang_b=self.target_lang_id,
        )

        # encoded_layers also includes the embedding layer
        # encoded_layers[-1] is the last layer
        last_layer, encoded_layers = self.xlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=lang_ids,
            position_ids=position_ids,
        )

        # TODO: select one of these strategies via cli
        # TODO: get a BETTER strategy
        # features = sum(encoded_layers[-5:-1])
        # features = encoded_layers[-2]
        features = last_layer

        if self.config.use_mlp:
            features = self.mlp(features)

        # Build the feature dictionary to be returned to the system
        output_features = self.split_outputs(
            features,
            batch_inputs,
            interleaved=self.config.interleave_input,
            label_a=const.SOURCE,
            label_b=const.TARGET,
        )

        # Convert pieces to tokens
        output_features[const.TARGET] = pieces_to_tokens(
            output_features[const.TARGET], batch_inputs[const.TARGET]
        )
        output_features[const.SOURCE] = pieces_to_tokens(
            output_features[const.SOURCE], batch_inputs[const.SOURCE]
        )
        source_len = batch_inputs[const.SOURCE].bounds_lengths
        target_len = batch_inputs[const.TARGET].bounds_lengths

        # NOTE: assuming here that features is already split into target and source
        source_features = output_features[const.SOURCE]
        target_features = output_features[const.TARGET]

        # Sentence-level features
        sentence_target_features = target_features[:, 0].unsqueeze(
            1
        ) + select_positions(target_features, (target_len - 1).unsqueeze(1))
        sentence_source_features = source_features[:, 0].unsqueeze(
            1
        ) + select_positions(source_features, (source_len - 1).unsqueeze(1))
        sentence_features = torch.cat(
            (sentence_target_features, sentence_source_features), dim=-1
        )

        output_features[const.TARGET_SENTENCE] = sentence_features
        output_features[const.TARGET] = target_features
        output_features[const.SOURCE] = source_features

        # Logits for multi-task fine-tuning
        if include_target_logits:
            output_features[const.TARGET_LOGITS] = torch.einsum(
                'vh,bsh->bsv',
                self.output_embeddings.weight,
                output_features[const.TARGET],
            )
        if include_source_logits:
            output_features[const.SOURCE_LOGITS] = torch.einsum(
                'vh,bsh->bsv',
                self.output_embeddings.weight,
                output_features[const.SOURCE],
            )

        # Additional features
        if self.config.use_mismatch_features:
            raise NotImplementedError

        return output_features

    @staticmethod
    def concat_input(batch_a, batch_b, pad_id, lang_a=None, lang_b=None):
        """Concatenate tensors of two batches into one tensor.

        Return:
            the concatenation, a mask of types (a as zeroes and b as ones)
                and concatenation of attention_mask.
        """
        ids_a = batch_a.tensor
        ids_b = batch_b.tensor
        attention_mask_a = retrieve_tokens_mask(batch_a)
        attention_mask_b = retrieve_tokens_mask(batch_b)
        types_a = torch.zeros_like(ids_a)
        types_b = torch.ones_like(ids_b)
        position_ids_a = torch.arange(
            ids_a.size(1), dtype=torch.long, device=ids_a.device
        )
        position_ids_a = position_ids_a.unsqueeze(0).expand(ids_a.size())
        position_ids_b = torch.arange(
            ids_b.size(1), dtype=torch.long, device=ids_b.device
        )
        position_ids_b = position_ids_b.unsqueeze(0).expand(ids_b.size())

        input_ids = torch.cat((ids_a, ids_b), dim=1)
        token_type_ids = torch.cat((types_a, types_b), dim=1)
        attention_mask = torch.cat((attention_mask_a, attention_mask_b), dim=1)
        position_ids = torch.cat((position_ids_a, position_ids_b), dim=1)

        if lang_a is not None and lang_b is not None:
            lang_id_a = torch.ones_like(ids_a) * lang_a
            lang_id_b = torch.ones_like(ids_b) * lang_b
            lang_ids = torch.cat((lang_id_a, lang_id_b), dim=1)
            # lang_ids *= attention_mask.unsqueeze(-1).to(lang_ids.dtype)
            lang_ids *= attention_mask.to(lang_ids.dtype)

            return input_ids, token_type_ids, attention_mask, position_ids, lang_ids

        return input_ids, token_type_ids, attention_mask, position_ids

    @staticmethod
    def interleave_input(batch_a, batch_b, pad_id, lang_a=None, lang_b=None):
        """Interleave the source + target embeddings into one tensor.

        This means making the input as [batch, target [SEP] source].

        Return:
            interleave of embds, mask of target (as zeroes) and source (as ones)
                and concatenation of attention_mask.
        """
        ids_a = batch_a.tensor
        ids_b = batch_b.tensor

        batch_size = ids_a.size(0)

        lengths_a = batch_a.lengths
        lengths_b = batch_b.lengths

        # max_pair_length = ids_a.size(1) + ids_b.size(1)
        max_pair_length = lengths_a + lengths_b

        input_ids = torch.full(
            (batch_size, max_pair_length),
            pad_id,
            dtype=ids_a.dtype,
            device=ids_a.device,
        )
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for i in range(batch_size):
            # <s> and </s> are included in the mask (=1)
            len_a = lengths_a[i].item()
            len_b = lengths_b[i].item()

            input_ids[i, :len_b] = ids_b[i, :len_b]
            token_type_ids[i, :len_b] = 0
            attention_mask[i, :len_b] = 1

            input_ids[i, len_b : len_b + len_a] = ids_a[i, :len_a]
            token_type_ids[i, len_b : len_b + len_a] = 1
            attention_mask[i, len_b : len_b + len_a] = 1

        # TODO, why is attention mask 1 for all positions?
        return input_ids, token_type_ids, attention_mask

    @staticmethod
    def split_outputs(
        features: torch.Tensor,
        batch_inputs,
        interleaved: bool = False,
        label_a: str = const.SOURCE,
        label_b: str = const.TARGET,
    ):
        """Split contexts to get tag_side outputs.

        Arguments:
            features (tensor): XLM output: <s> source </s> </s> target </s>
                Shape of (bs, 1 + source_len + 2 + target_len + 1, 2)
            batch_inputs:
            interleaved (bool): whether the concat strategy was 'interleaved'.
            label_a: dictionary key for sequence A in ``features``.
            label_b: dictionary key for sequence B in ``features``.

        Return:
            dict of tensors, one per tag side.
        """
        outputs = OrderedDict()

        if interleaved:
            raise NotImplementedError('interleaving not supported.')
            # TODO: fix code below to use the lengths information and not bounds
            # if interleaved, shift each source sample by its correspondent length
            lengths_a = batch_inputs[const.TARGET].lengths
            shift = lengths_a.unsqueeze(-1)

            range_vector = torch.arange(
                features.size(0), device=features.device
            ).unsqueeze(1)

            target_bounds = batch_inputs[const.TARGET].bounds
            features_a = features[range_vector, target_bounds]
            # Shift bounds by target length and preserve padding
            source_bounds = batch_inputs[const.SOURCE].bounds
            m = (source_bounds != -1).long()  # for masking out padding (which is -1)
            shifted_bounds = (source_bounds + shift) * m + source_bounds * (1 - m)
            features_b = features[range_vector, shifted_bounds]
        else:
            # otherwise, shift all by max_length
            lengths_a = batch_inputs[label_a].lengths
            # if we'd like to maintain the word pieces we merely select all
            features_a = features[:, : lengths_a.max()]
            features_b = features[:, lengths_a.max() :]

        outputs[label_a] = features_a
        outputs[label_b] = features_b

        return outputs
