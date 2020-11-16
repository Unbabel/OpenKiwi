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
from typing import Dict, List, Union

import torch
from pydantic import DirectoryPath, confloat
from pydantic.class_validators import validator
from torch import Tensor, nn
from transformers import (
    XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    AdapterType,
    AutoTokenizer,
    XLMRobertaConfig,
    XLMRobertaModel,
)
from transformers.adapter_config import PfeifferConfig
from typing_extensions import Literal

from kiwi import constants as const
from kiwi.data.encoders.field_encoders import TextEncoder
from kiwi.data.vocabulary import Vocabulary
from kiwi.modules.common.scalar_mix import ScalarMixWithDropout
from kiwi.systems._meta_module import MetaModule
from kiwi.utils.data_structures import DefaultFrozenDict
from kiwi.utils.io import BaseConfig
from kiwi.utils.tensors import pieces_to_tokens, retrieve_tokens_mask

logger = logging.getLogger(__name__)


class XLMRobertaTextEncoder(TextEncoder):
    def __init__(self, tokenizer_name='xlm-roberta-base', is_source=False):
        if tokenizer_name not in XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST:
            tokenizer_name = 'xlm-roberta-base'

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name))
        wordpiece_tokenize = tokenizer._tokenize

        # init_token = None if is_source else xlmroberta_tokenizer.cls_token

        super().__init__(
            subtokenize=wordpiece_tokenize,
            pad_token=tokenizer.pad_token,
            unk_token=tokenizer.unk_token,
            bos_token=tokenizer.cls_token,
            eos_token=tokenizer.eos_token,
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

        vocab = tokenizer.get_vocab()
        self.vocab.stoi = DefaultFrozenDict(vocab, vocab[self.unk_token])

        inverted_vocab = {v: k for k, v in self.vocab.stoi.items()}
        self.vocab.itos = DefaultFrozenDict(
            inverted_vocab, inverted_vocab[vocab[self.unk_token]]
        )

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
            'Vocabulary already defined for XLMRoberta field; '
            'not fitting it to data now'
        )
        if vocab_size:
            self.vocab.max_size(vocab_size)


class EncoderAdapterConfig(BaseConfig):
    language: str = None
    """Specify a name to add a new language adapter, e.g. 'en-de', 'en-zh', etc."""

    load: List[DirectoryPath] = None
    """Load trained adapters to use for prediction or for Adapter Fusion.
    Point it to the root directory."""

    fusion: bool = False
    """Train Adapter Fusion on top of the loaded adapters."""

    @validator('fusion')
    def check_load(cls, v, values):
        if v and not values.get('load'):
            raise NotImplementedError(
                'Specify adapters to load if you want to fuse them'
            )
        return v


@MetaModule.register_subclass
class XLMRobertaEncoder(MetaModule):
    """XLM-RoBERTa model, using HuggingFace's implementation."""

    class Config(BaseConfig):
        model_name: Union[str, Path] = 'xlm-roberta-base'
        """Pre-trained XLMRoberta model to use."""

        adapter: EncoderAdapterConfig = None
        """Use an Adapter to fine tune the encoder."""

        interleave_input: bool = False
        """Concatenate SOURCE and TARGET without internal padding
        (111222000 instead of 111002220)"""

        use_mlp: bool = True
        """Apply a linear layer on top of XLMRoberta."""

        hidden_size: int = 100
        """Size of the linear layer on top of XLMRoberta."""

        pooling: Literal['first_token', 'mean', 'll_mean', 'mixed'] = 'mixed'
        """Type of pooling used to extract features from the encoder. Options are:
            first_token: CLS_token is used for sentence representation
            mean: Use avg pooling for sentence representation using scalar mixed layers
            ll_mean: Mean pool of only last layer embeddings
            mixed: Concat CLS token with mean_pool"""

        scalar_mix_dropout: confloat(ge=0.0, le=1.0) = 0.1
        scalar_mix_layer_norm: bool = True

        freeze: bool = False
        """Freeze XLMRoberta during training."""

        freeze_for_number_of_steps: int = 0
        """Freeze XLMR during training for this number of steps."""

        @validator('model_name', pre=True)
        def fix_relative_path(cls, v):
            if v not in XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST:
                v = Path(v)
                if not v.is_absolute():
                    v = Path.cwd().joinpath(v)
            return v

    def __init__(
        self, vocabs: Dict[str, Vocabulary], config: Config, pre_load_model: bool = True,
    ):
        super().__init__(config=config)

        if pre_load_model:
            self.xlm_roberta = XLMRobertaModel.from_pretrained(
                self.config.model_name, output_hidden_states=True
            )
        else:
            xlm_roberta_config = XLMRobertaConfig.from_pretrained(
                self.config.model_name, output_hidden_states=True
            )
            self.xlm_roberta = XLMRobertaModel(xlm_roberta_config)

        # Add Adapters if specified
        if config.adapter is not None:
            if config.adapter.language is not None:
                # Add an adapter module
                self.xlm_roberta.add_adapter(
                    config.adapter.language,
                    AdapterType.text_lang,
                    config=PfeifferConfig(),
                )
                self.xlm_roberta.train_adapter(config.adapter.language)
            if config.adapter.load is not None:
                # Load the adapter module
                for path in config.adapter.load:
                    self.xlm_roberta.load_adapter(
                        str(path), AdapterType.text_lang, config=PfeifferConfig(),
                    )
            # Add fusion of adapters
            if config.adapter.fusion:
                adapter_setup = [[path.name for path in config.adapter.load]]
                self.xlm_roberta.add_fusion(adapter_setup[0], "dynamic")
                self.xlm_roberta.train_fusion(adapter_setup)

        self.vocabs = {
            const.TARGET: vocabs[const.TARGET],
            const.SOURCE: vocabs[const.SOURCE],
        }

        self.mlp = None

        if self.config.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.xlm_roberta.config.hidden_size, self.config.hidden_size),
                nn.Tanh(),
            )
            output_size = self.config.hidden_size
        else:
            output_size = self.xlm_roberta.config.hidden_size

        sentence_size = output_size
        if config.pooling == 'mixed':
            sentence_size *= 2

        self.scalar_mix = ScalarMixWithDropout(
            mixture_size=self.xlm_roberta.config.num_hidden_layers
            + 1,  # +1 for embeddings
            do_layer_norm=self.config.scalar_mix_layer_norm,
            dropout=self.config.scalar_mix_dropout,
        )

        self._sizes = {
            const.TARGET: output_size,
            const.TARGET_LOGITS: output_size,
            const.TARGET_SENTENCE: sentence_size,
            const.SOURCE: output_size,
        }

        self.output_embeddings = self.xlm_roberta.embeddings.word_embeddings

        self._training_steps_ran = 0
        self._is_frozen = False
        if self.config.freeze:
            logger.info(
                'Freezing XLMRoberta encoder weights; training will not update them'
            )
            for param in self.xlm_roberta.parameters():
                param.requires_grad = False
            self._is_frozen = True
        if self.config.freeze_for_number_of_steps > 0:
            # Done inside `forward()` to guarantee we can unfreeze (if optimizer is
            #  built after this, we cannot unfreeze without calling
            #  `optimizer.add_param_group({'params': self.xlm.parameters()})`
            pass

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
                self.xlm_roberta.embeddings._non_persistent_buffers_set.add(
                    'position_ids'
                )
                keys = super().load_state_dict(state_dict, strict)
                self.xlm_roberta.embeddings._non_persistent_buffers_set.discard(
                    'position_ids'
                )
            else:
                raise e
        return keys

    @classmethod
    def input_data_encoders(cls, config: Config):
        return {
            const.SOURCE: XLMRobertaTextEncoder(tokenizer_name=config.model_name),
            const.TARGET: XLMRobertaTextEncoder(tokenizer_name=config.model_name),
        }

    def size(self, field=None):
        if field:
            return self._sizes[field]
        return self._sizes

    def _check_freezing(self):
        if self._training_steps_ran == 0 and self.config.freeze_for_number_of_steps > 0:
            logger.info(
                f'Freezing XLMRoberta encoder weights for '
                f'{self.config.freeze_for_number_of_steps} steps'
            )
            for param in self.xlm_roberta.parameters():
                param.requires_grad = False
            self._is_frozen = True
        elif (
            self._is_frozen
            and self._training_steps_ran >= self.config.freeze_for_number_of_steps
        ):
            logger.info(
                f'Unfreezing XLMRoberta encoder '
                f'({self._training_steps_ran} steps have passed)'
            )
            for param in self.xlm_roberta.parameters():
                param.requires_grad = True
            self._is_frozen = False

        self._training_steps_ran += 1

    def forward(self, batch_inputs, *args, include_logits=False):
        self._check_freezing()

        # Input is a concatenation of both embeddings or an interleave
        if self.config.interleave_input:
            merge_input_fn = self.interleave_input
        else:
            merge_input_fn = self.concat_input

        input_ids, token_type_ids, attention_mask = merge_input_fn(
            batch_inputs[const.SOURCE],
            batch_inputs[const.TARGET],
            pad_id=self.vocabs[const.TARGET].pad_id,
        )

        # encoded_layers also includes the embedding layer
        # encoded_layers[-1] is the last layer
        # pooled_output is the first token of the last layer [CLS]
        last_layer, pooled_output, encoded_layers = self.xlm_roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        features = self.scalar_mix(encoded_layers, attention_mask)

        if self.config.use_mlp:
            features = self.mlp(features)
            last_layer = self.mlp(last_layer)
            pooled_output = self.mlp(pooled_output)

        # Build the feature dictionary to be returned to the system
        split_features = self.split_outputs(
            features, batch_inputs, interleaved=self.config.interleave_input
        )
        split_last_layer = self.split_outputs(
            last_layer, batch_inputs, interleaved=self.config.interleave_input
        )

        mask = retrieve_tokens_mask(batch_inputs[const.TARGET])

        average_pooling = (split_features[const.TARGET] * mask[:, :, None]).sum(
            1
        ) / mask.sum(1)[:, None]

        last_layer_average_pooling = (
            split_last_layer[const.TARGET] * mask[:, :, None]
        ).sum(1) / mask.sum(1)[:, None]

        # interesting idea to try, but not what we implemented
        # mixed_pool = torch.cat((average_pooling, last_layer_average_pooling), 1)
        mixed_pool = torch.cat((average_pooling, pooled_output), 1)

        # Pooling the token embeddings that have already been scalar mixed
        # (mean of tokens)
        sentence_features = pooled_output
        if self.config.pooling == 'mixed':
            sentence_features = mixed_pool
        elif self.config.pooling == 'mean':
            sentence_features = average_pooling
        elif self.config.pooling == 'll_mean':
            sentence_features = last_layer_average_pooling
        elif self.config.pooling == 'first_token':
            sentence_features = pooled_output

        # Convert pieces to tokens
        output_features = {
            const.TARGET: pieces_to_tokens(
                split_features[const.TARGET], batch_inputs[const.TARGET]
            ),
            const.SOURCE: pieces_to_tokens(
                split_features[const.SOURCE], batch_inputs[const.SOURCE]
            ),
            const.TARGET_SENTENCE: sentence_features,
        }

        # Logits for multi-task fine-tuning
        if include_logits:
            # FIXME: this is wrong
            raise NotImplementedError('Logic not implemented for the XLMR encoder.')

        return output_features

    @staticmethod
    def concat_input(source_batch, target_batch, pad_id):
        """Concatenate tensors of two batches into one tensor.

        Return:
            the concatenation, a mask of types (a as zeroes and b as ones)
                and concatenation of attention_mask.
        """
        source_ids = source_batch.tensor
        target_ids = target_batch.tensor

        source_attention_mask = retrieve_tokens_mask(source_batch)
        target_attention_mask = retrieve_tokens_mask(target_batch)

        input_ids = torch.cat((target_ids, source_ids), dim=1)

        # XLMR does not use NSP
        token_type_ids = torch.zeros_like(input_ids)

        attention_mask = torch.cat(
            (target_attention_mask, source_attention_mask), dim=1
        )
        return input_ids, token_type_ids, attention_mask

    @staticmethod
    def split_outputs(features, batch_inputs, interleaved=False):
        """Split contexts to get tag_side outputs.

        Arguments:
            features (tensor): XLMRoberta output: <s> target </s> </s> source </s>
                Shape of (bs, 1 + target_len + 2 + source_len + 1, 2)
            batch_inputs:
            interleaved (bool): whether the concat strategy was 'interleaved'.

        Return:
            dict of tensors, one per tag side.
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

    @staticmethod
    def interleave_input(source_batch, target_batch, pad_id):
        """Interleave the source + target embeddings into one tensor.

        This means making the input as [batch, target [SEP] source].

        Return:
            interleave of embds, mask of target (as zeroes) and source (as ones)
                and concatenation of attention_mask
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
            # XLMR does not use NSP
            # token_type_ids[i, t_len : t_len + s_len] = 1
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
