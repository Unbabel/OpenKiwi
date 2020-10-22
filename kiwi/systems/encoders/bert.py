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
from pydantic import confloat
from pydantic.class_validators import validator
from torch import Tensor, nn
from transformers import (
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    AutoTokenizer,
    BertConfig,
    BertModel,
)

from kiwi import constants as const
from kiwi.data.batch import MultiFieldBatch
from kiwi.data.encoders.field_encoders import TextEncoder
from kiwi.data.vocabulary import Vocabulary
from kiwi.modules.common.scalar_mix import ScalarMixWithDropout
from kiwi.systems._meta_module import MetaModule
from kiwi.utils.data_structures import DefaultFrozenDict
from kiwi.utils.io import BaseConfig
from kiwi.utils.tensors import pieces_to_tokens, retrieve_tokens_mask

logger = logging.getLogger(__name__)


class TransformersTextEncoder(TextEncoder):
    def __init__(self, tokenizer_name, is_source=False):
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name))
        wordpiece_tokenize = tokenizer.wordpiece_tokenizer.tokenize

        init_token = None if is_source else tokenizer.cls_token

        super().__init__(
            subtokenize=wordpiece_tokenize,
            pad_token=tokenizer.pad_token,
            unk_token=tokenizer.unk_token,
            bos_token=init_token,
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
            tokenizer.vocab, tokenizer.vocab[self.unk_token]
        )
        self.vocab.itos = tokenizer.ids_to_tokens

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
            'Vocabulary already defined for Bert field; not fitting it to data now'
        )


@MetaModule.register_subclass
class BertEncoder(MetaModule):
    """BERT model as presented in Google's paper and using Hugging Face's code

    References:
        https://arxiv.org/abs/1810.04805
    """

    class Config(BaseConfig):
        encode_source: bool = False

        model_name: Union[str, Path] = 'bert-base-multilingual-cased'
        """Pre-trained BERT model to use."""

        use_mismatch_features: bool = False
        """Use Alibaba's mismatch features."""

        use_predictor_features: bool = False
        """Use features originally proposed in the Predictor model."""

        interleave_input: bool = False
        """Concatenate SOURCE and TARGET without internal padding
        (111222000 instead of 111002220)"""

        freeze: bool = False
        """Freeze BERT during training."""

        use_mlp: bool = True
        """Apply a linear layer on top of BERT."""

        hidden_size: int = 100
        """Size of the linear layer on top of BERT."""

        scalar_mix_dropout: confloat(ge=0.0, le=1.0) = 0.1
        scalar_mix_layer_norm: bool = True

        @validator('model_name', pre=True)
        def fix_relative_path(cls, v):
            if (
                v not in BERT_PRETRAINED_MODEL_ARCHIVE_LIST
                and v not in DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
            ):
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
            self.bert = BertModel.from_pretrained(
                self.config.model_name, output_hidden_states=True
            )
        else:
            bert_config = BertConfig.from_pretrained(
                self.config.model_name, output_hidden_states=True
            )
            self.bert = BertModel(bert_config)

        self.vocabs = {
            const.TARGET: vocabs[const.TARGET],
            const.SOURCE: vocabs[const.SOURCE],
        }

        self.mlp = None
        if self.config.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, self.config.hidden_size),
                nn.Tanh(),
            )
            output_size = self.config.hidden_size
        else:
            output_size = self.bert.config.hidden_size

        self.scalar_mix = ScalarMixWithDropout(
            mixture_size=self.bert.config.num_hidden_layers + 1,  # +1 for embeddings
            do_layer_norm=self.config.scalar_mix_layer_norm,
            dropout=self.config.scalar_mix_dropout,
        )

        self._sizes = {
            const.TARGET: output_size,
            const.TARGET_LOGITS: output_size,
            const.TARGET_SENTENCE: self.bert.config.hidden_size,
            const.SOURCE: output_size,
        }

        self.output_embeddings = self.bert.embeddings.word_embeddings

        if self.config.freeze:
            for param in self.bert.parameters():
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
                self.bert.embeddings._non_persistent_buffers_set.add('position_ids')
                keys = super().load_state_dict(state_dict, strict)
                self.bert.embeddings._non_persistent_buffers_set.discard('position_ids')
            else:
                raise e
        return keys

    @classmethod
    def input_data_encoders(cls, config: Config):
        return {
            const.SOURCE: TransformersTextEncoder(
                tokenizer_name=config.model_name, is_source=True
            ),
            const.TARGET: TransformersTextEncoder(tokenizer_name=config.model_name),
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
        include_source_logits=False
    ):
        # BERT gets it's input as a concatenation of both embeddings
        # or as an interleave of inputs
        if self.config.interleave_input:
            merge_input_fn = self.interleave_input
        else:
            merge_input_fn = self.concat_input

        input_ids, token_type_ids, attention_mask = merge_input_fn(
            batch_inputs[const.SOURCE],
            batch_inputs[const.TARGET],
            pad_id=self.vocabs[const.TARGET].pad_id,
        )

        # hidden_states also includes the embedding layer
        # hidden_states[-1] is the last layer
        last_hidden_state, pooler_output, hidden_states = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # TODO: select one of these strategies via cli
        # TODO: get a BETTER strategy
        features = self.scalar_mix(hidden_states, attention_mask)

        # features = sum(hidden_states[-5:-1])
        # features = hidden_states[-2]

        if self.config.use_mlp:
            features = self.mlp(features)

        # Build the feature dictionary to be returned to the system
        output_features = self.split_outputs(
            features, batch_inputs, interleaved=self.config.interleave_input
        )

        # Convert pieces to tokens
        target_features = pieces_to_tokens(
            output_features[const.TARGET], batch_inputs[const.TARGET]
        )
        source_features = pieces_to_tokens(
            output_features[const.SOURCE], batch_inputs[const.SOURCE]
        )

        # sentence_features = pooler_output
        sentence_features = last_hidden_state.mean(dim=1)

        # Substitute CLS on target side
        # target_features[:, 0] = 0

        output_features[const.TARGET] = target_features
        output_features[const.SOURCE] = source_features
        output_features[const.TARGET_SENTENCE] = sentence_features

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
    def concat_input(source_batch, target_batch, pad_id):
        """Concatenate the target + source embeddings into one tensor.

        Return:
             concatenation of embeddings, mask of target (as ones) and source
                 (as zeroes) and concatenation of attention_mask
        """
        source_ids = source_batch.tensor
        target_ids = target_batch.tensor

        source_attention_mask = retrieve_tokens_mask(source_batch)
        target_attention_mask = retrieve_tokens_mask(target_batch)

        target_types = torch.zeros_like(target_ids)
        # zero denotes first sequence
        source_types = torch.ones_like(source_ids)
        input_ids = torch.cat((target_ids, source_ids), dim=1)
        token_type_ids = torch.cat((target_types, source_types), dim=1)
        attention_mask = torch.cat(
            (target_attention_mask, source_attention_mask), dim=1
        )
        return input_ids, token_type_ids, attention_mask

    @staticmethod
    def split_outputs(
        features: Tensor, batch_inputs: MultiFieldBatch, interleaved: bool = False
    ) -> Dict[str, Tensor]:
        """Split features back into sentences A and B.

        Args:
            features: BERT's output: ``[CLS] target [SEP] source [SEP]``.
                Shape of (bs, 1 + target_len + 1 + source_len + 1, 2)
            batch_inputs: the regular batch object, containing ``source`` and ``target``
                batches
            interleaved: whether the concat strategy was interleaved

        Return:
            dict of tensors for ``source`` and ``target``.
        """
        outputs = OrderedDict()

        target_lengths = batch_inputs[const.TARGET].lengths

        if interleaved:
            raise NotImplementedError('interleaving not supported.')
            # TODO: fix code below to use the lengths information and not bounds
            # if interleaved, shift each source sample by its correspondent length
            shift = target_lengths.unsqueeze(-1)

            range_vector = torch.arange(
                features.size(0), device=features.device
            ).unsqueeze(1)

            target_bounds = batch_inputs[const.TARGET].bounds
            target_features = features[range_vector, target_bounds]
            # Shift bounds by target length and preserve padding
            source_bounds = batch_inputs[const.SOURCE].bounds
            m = (source_bounds != -1).long()  # for masking out padding (which is -1)
            shifted_bounds = (source_bounds + shift) * m + source_bounds * (1 - m)
            source_features = features[range_vector, shifted_bounds]
        else:
            # otherwise, shift all by max_length
            # if we'd like to maintain the word pieces we merely select all
            target_features = features[:, : target_lengths.max()]
            # ignore the target and get the rest
            source_features = features[:, target_lengths.max() :]

        outputs[const.TARGET] = target_features

        # Source doesn't have an init_token (like CLS) and we keep SEP
        outputs[const.SOURCE] = source_features

        return outputs

    # TODO this strategy is not being used, should we keep it?
    @staticmethod
    def interleave_input(source_batch, target_batch, pad_id):
        """Interleave the source + target embeddings into one tensor.

        This means making the input as [batch, target [SEP] source].

        Return:
            interleave of embds, mask of target (as zeroes) and source (as ones)
                and concatenation of attention_mask.
        """
        source_ids = source_batch.tensor
        target_ids = target_batch.tensor

        batch_size = source_ids.size(0)

        source_lengths = source_batch.lengths
        target_lengths = target_batch.lengths

        max_pair_length = source_ids.size(1) + target_ids.size(1)

        input_ids = torch.full(
            (batch_size, max_pair_length),
            pad_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for i in range(batch_size):
            # [CLS] and [SEP] are included in the mask (=1)
            # note: source does not have CLS
            t_len = target_lengths[i].item()
            s_len = source_lengths[i].item()

            input_ids[i, :t_len] = target_ids[i, :t_len]
            token_type_ids[i, :t_len] = 0
            attention_mask[i, :t_len] = 1

            input_ids[i, t_len : t_len + s_len] = source_ids[i, :s_len]
            token_type_ids[i, t_len : t_len + s_len] = 1
            attention_mask[i, t_len : t_len + s_len] = 1

        # TODO, why is attention mask 1 for all positions?
        return input_ids, token_type_ids, attention_mask

    @staticmethod
    def get_mismatch_features(logits, target, pred):
        # calculate mismatch features and concat them
        t_max = torch.gather(logits, -1, target.unsqueeze(-1))
        p_max = torch.gather(logits, -1, pred.unsqueeze(-1))
        diff_max = t_max - p_max
        diff_arg = (target != pred).float().unsqueeze(-1)
        mismatch = torch.cat((t_max, p_max, diff_max, diff_arg), dim=-1)
        return mismatch
