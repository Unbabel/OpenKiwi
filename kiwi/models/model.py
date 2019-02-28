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

import logging
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

import kiwi
from kiwi import constants as const
from kiwi.data import utils
from kiwi.data.vocabulary import Vocabulary
logger = logging.getLogger(__name__)


class ModelConfig(object):
    __metaclass__ = ABCMeta

    def __init__(self, vocabs, **kwargs):
        """Model Configuration Base Class.

        Args:
        vocabs: Dictionary Mapping Field Names to Vocabularies.
                Must contain 'source' and 'target' keys
        """
        super().__init__(**kwargs)
        self.pad_idx = {}
        self.stop_idx = {}
        self.start_idx = {}
        self.vocab_sizes = {}

        for side in [const.SOURCE, const.TARGET, const.PE]:
            if side in vocabs:
                self.pad_idx[side] = vocabs[side].token_to_id(const.PAD)
                self.stop_idx[side] = vocabs[side].token_to_id(const.STOP)
                self.start_idx[side] = vocabs[side].token_to_id(const.START)
                self.vocab_sizes[side] = len(vocabs[side])

    def update(self, other_config):
        """Updates the config object with the values of `other_config`
           Args:
             other_config: The `dict` or `ModelConfig` object to update with.
        """
        config_dict = dict()
        if isinstance(self, other_config.__class__):
            config_dict = other_config.__dict__
        elif isinstance(other_config, dict):
            config_dict = other_config
        self.__dict__.update(config_dict)

    def state_dict(self):
        """Return the __dict__ for serialization.
        """
        self.__dict__['__version__'] = kiwi.__version__
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict, vocabs, **kwargs):
        """Create config from a saved state_dict.
           Args:
             config_dict: A dictionary that is the return value of
                          a call to the `state_dict()` method of `cls`
             vocab: See `ModelConfig.__init__`
             kwargs: keyword arguments that are passed to the `cls` constructor
        """
        config_dict, kwargs = cls.convert_serial_format(config_dict, kwargs)
        config = cls(vocabs=vocabs, **kwargs)
        config.update(config_dict)
        return config

    @staticmethod
    def convert_serial_format(config_dict, kwargs):
        """Currently does nothing.
        """
        return config_dict, kwargs


class QEModelConfig(ModelConfig):
    """Config for a Quality Estimation Model.
    """

    def __init__(
            self,
            vocabs,
            predict_target,
            predict_source,
            predict_gaps,
            target_bad_weight=2.0,
            source_bad_weight=2.0,
            gaps_bad_weight=2.0,
            **kwargs
    ):
        super().__init__(vocabs, **kwargs)
        self.bad_weights = {}
        self.output_tags = []
        if predict_target:
            self.bad_weights[const.TARGET_TAGS] = target_bad_weight
            self.output_tags.append(const.TARGET_TAGS)
        if predict_source:
            self.bad_weights[const.SOURCE_TAGS] = source_bad_weight
            self.output_tags.append(const.SOURCE_TAGS)
        if predict_gaps:
            self.bad_weights[const.GAP_TAGS] = gaps_bad_weight
            self.output_tags.append(const.GAP_TAGS)

        self.bad_idx = {}
        self.nb_classes = {}
        for tag in self.output_tags:
            self.pad_idx[tag] = vocabs[tag].token_to_id(const.PAD)
            self.bad_idx[tag] = vocabs[tag].token_to_id(const.BAD)
            self.nb_classes[tag] = len(vocabs[tag]) - 1  # Ignore Pad

    @staticmethod
    def convert_serial_format(config_dict, kwargs):
        config_dict, kwargs = (super(QEModelConfig, QEModelConfig)
                               .convert_serial_format(config_dict, kwargs))
        if '__version__' not in config_dict:
            kwargs['predict_target'] = config_dict['predict_target']
            kwargs['predict_source'] = config_dict['predict_source']
            kwargs['predict_gaps'] = config_dict['predict_gaps']
        return config_dict, kwargs


class Model(nn.Module):
    __metaclass__ = ABCMeta

    subclasses = {}

    def __init__(self, vocabs, ConfigCls=ModelConfig, config=None, **kwargs):
        """Quality Estimation Base Class.

        Args:
            vocabs: Dictionary Mapping Field Names to Vocabularies.
            ConfigCls: ModelConfig Subclass
            config: A State Dict of a ModelConfig subclass.
                    If set, passing other kwargs will raise an Exception.
        """

        super().__init__()

        self.vocabs = vocabs

        if config is None:
            config = ConfigCls(vocabs=vocabs, **kwargs)
        else:
            config = ConfigCls.from_dict(config_dict=config, vocabs=vocabs)
            assert not kwargs
        self.config = config

    @classmethod
    def register_subclass(cls, subclass):
        cls.subclasses[subclass.__name__] = subclass
        return subclass

    @abstractmethod
    def loss(self, model_out, target):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def predict(self, batch, class_name=const.BAD, unmask=True):
        model_out = self(batch)
        predictions = {}

        for key in model_out:
            if key in [const.TARGET_TAGS, const.SOURCE_TAGS, const.GAP_TAGS]:
                # Models are assumed to return logits, not probabilities
                logits = model_out[key]
                probs = torch.softmax(logits, dim=-1)
                class_index = torch.LongTensor(
                    [self.vocabs[key].token_to_id(class_name)],
                    device=probs.device,
                )
                class_probs = probs.index_select(-1, class_index)
                class_probs = class_probs.squeeze(-1).tolist()
                if unmask:
                    if key == const.SOURCE_TAGS:
                        input_key = const.SOURCE
                    else:
                        input_key = const.TARGET
                    mask = self.get_mask(batch, input_key)
                    if key == const.GAP_TAGS:
                        # Append one extra token
                        mask = torch.cat(
                            [mask.new_ones((mask.shape[0], 1)), mask], dim=1
                        )

                    lengths = mask.int().sum(dim=-1)
                    for i, x in enumerate(class_probs):
                        class_probs[i] = x[: lengths[i]]
                predictions[key] = class_probs
            elif key == const.SENTENCE_SCORES:
                predictions[key] = model_out[key].tolist()
            elif key == const.BINARY:
                logits = model_out[key]
                probs = torch.softmax(logits, dim=-1)
                class_probs = probs[..., 0]
                predictions[key] = class_probs.tolist()

        return predictions

    def predict_raw(self, examples):
        batch = self.preprocess(examples)
        return self.predict(batch, class_name=const.BAD, unmask=True)

    def preprocess(self, examples):
        """Preprocess Raw Data.

        Args:
            examples (list of dict): List of examples. Each Example is a dict
                                     with field strings as keys, and
                                     unnumericalized, tokenized data as values.
        Return:
            A batch object.
        """
        raise NotImplementedError

    def get_mask(self, batch, output):
        """Compute Mask of Tokens for side.

        Args:
            batch: Namespace of tensors
            side: String identifier.
        """
        side = output
        # if output in [const.TARGET_TAGS, const.GAP_TAGS]:
        #     side = const.TARGET
        # elif output == const.SOURCE_TAGS:
        #     side = const.SOURCE

        input_tensor = getattr(batch, side)
        if isinstance(input_tensor, tuple) and len(input_tensor) == 2:
            input_tensor, lengths = input_tensor

        # output_tensor = getattr(batch, output)
        # if isinstance(output_tensor, tuple) and len(output_tensor) == 2:
        #     output_tensor, lengths = output_tensor

        mask = torch.ones_like(input_tensor, dtype=torch.uint8)

        possible_padding = [const.PAD, const.START, const.STOP]

        unk_id = self.vocabs[side].stoi.get(const.UNK)
        for pad in possible_padding:
            pad_id = self.vocabs[side].stoi.get(pad)
            if pad_id is not None and pad_id != unk_id:
                mask &= input_tensor != pad_id

        return mask

    @staticmethod
    def create_from_file(path):
        model_dict = torch.load(
            str(path), map_location=lambda storage, loc: storage
        )
        for model_name in Model.subclasses:
            if model_name in model_dict:
                model = Model.subclasses[model_name].from_dict(model_dict)
                return model
        return None

    @classmethod
    def from_file(cls, path):
        model_dict = torch.load(
            str(path), map_location=lambda storage, loc: storage
        )
        if cls.__name__ not in model_dict:
            raise KeyError(
                '{} model data not found in {}'.format(cls.__name__, path)
            )

        return cls.from_dict(model_dict)

    @classmethod
    def from_dict(cls, class_dict):
        if '__version__' not in class_dict:
            class_dict.update(class_dict[cls.__name__])
            del class_dict[cls.__name__]
        class_dict = cls.convert_serial_format(class_dict)
        vocabs = dict(class_dict[const.VOCAB])
        model = cls(vocabs=vocabs, config=class_dict[const.CONFIG])
        model.load_state_dict(class_dict[const.STATE_DICT])
        return model

    @staticmethod
    def convert_serial_format(class_dict):
        if '__version__' not in class_dict:
            vocabs = dict(class_dict[const.VOCAB])
            class_dict = class_dict
            for key in vocabs.keys():
                vocabs[key] = Vocabulary.from_vocab(vocabs[key])
            class_dict[const.VOCAB] = vocabs
        return class_dict

    def save(self, path):
        vocabs = utils.serialize_vocabs(self.vocabs)
        model_dict = {
            '__version__': kiwi.__version__,
            'class_name': self.__class__.__name__,
            const.VOCAB: vocabs,
            const.CONFIG: self.config.state_dict(),
            const.STATE_DICT: self.state_dict(),
        }
        torch.save(model_dict, str(path))
