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
from kiwi.models.utils import load_torch_file

logger = logging.getLogger(__name__)


class ModelConfig:
    __metaclass__ = ABCMeta

    def __init__(self, vocabs):
        """Model Configuration Base Class.

        Args:
        vocabs: Dictionary Mapping Field Names to Vocabularies.
                Must contain 'source' and 'target' keys
        """
        self.source_vocab_size = len(vocabs[const.SOURCE])
        self.target_vocab_size = len(vocabs[const.TARGET])

    @classmethod
    def from_dict(cls, config_dict, vocabs):
        """Create config from a saved state_dict.
           Args:
             config_dict: A dictionary that is the return value of
                          a call to the `state_dict()` method of `cls`
             vocab: See `ModelConfig.__init__`
        """
        config = cls(vocabs)
        config.update(config_dict)
        return config

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
        class_index = torch.tensor([const.LABELS.index(class_name)])

        for key in model_out:
            if key in [const.TARGET_TAGS, const.SOURCE_TAGS, const.GAP_TAGS]:
                # Models are assumed to return logits, not probabilities
                logits = model_out[key]
                probs = torch.softmax(logits, dim=-1)
                class_probs = probs.index_select(
                    -1, class_index.to(device=probs.device)
                )
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
                class_probs = probs.index_select(
                    -1, class_index.to(device=probs.device)
                )
                predictions[key] = class_probs.tolist()

        return predictions

    def predict_raw(self, examples):
        batch = self.preprocess(examples)
        return self.predict(batch, class_name=const.BAD_ID, unmask=True)

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
                mask &= torch.as_tensor(
                    input_tensor != pad_id,
                    device=mask.device,
                    dtype=torch.uint8,
                )

        return mask

    @staticmethod
    def create_from_file(path):

        try:
            model_dict = load_torch_file(path)
        except FileNotFoundError:
            # If no model is found
            raise FileNotFoundError(
                'No valid model data found in {}'.format(path)
            )

        for model_name in Model.subclasses:
            if model_name in model_dict:
                model = Model.subclasses[model_name].from_dict(model_dict)
                return model

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
    def from_dict(cls, model_dict):
        vocabs = utils.deserialize_vocabs(model_dict[const.VOCAB])
        class_dict = model_dict[cls.__name__]
        model = cls(vocabs=vocabs, config=class_dict[const.CONFIG])
        model.load_state_dict(class_dict[const.STATE_DICT])
        return model

    def save(self, path):
        vocabs = utils.serialize_vocabs(self.vocabs)
        model_dict = {
            '__version__': kiwi.__version__,
            const.VOCAB: vocabs,
            self.__class__.__name__: {
                const.CONFIG: self.config.state_dict(),
                const.STATE_DICT: self.state_dict(),
            },
        }
        torch.save(model_dict, str(path))
