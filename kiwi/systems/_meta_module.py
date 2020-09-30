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
import json
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn

import kiwi
from kiwi import constants as const
from kiwi.utils.io import BaseConfig, load_torch_file

logger = logging.getLogger(__name__)


class Serializable(metaclass=ABCMeta):
    subclasses = {}

    @classmethod
    def register_subclass(cls, subclass):
        cls.subclasses[subclass.__name__] = subclass
        return subclass

    @classmethod
    def retrieve_subclass(cls, subclass_name):
        subclass = cls.subclasses.get(subclass_name)
        if subclass is None:
            raise KeyError(
                f'{subclass_name} is not a registered subclass of {cls.__name__}'
            )
        return subclass

    @classmethod
    def load(cls, path):
        model_dict = load_torch_file(path)
        return cls.from_dict(model_dict)

    def save(self, path):
        torch.save(self.to_dict(), path)

    @classmethod
    @abstractmethod
    def from_dict(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def to_dict(cls, include_state=True):
        pass


class MetaModule(nn.Module, Serializable, metaclass=ABCMeta):
    class Config(BaseConfig, metaclass=ABCMeta):
        pass

    def __init__(self, config: Config):
        """Base module used for several model layers and modules.

        Arguments:
            config: a ``MetaModule.Config`` object.
        """
        super().__init__()

        self.config = config

    @classmethod
    def from_dict(cls, module_dict, **kwargs):
        module_cls = cls.retrieve_subclass(module_dict['class_name'])
        config = module_cls.Config(**module_dict[const.CONFIG])
        module = module_cls(config=config, **kwargs)

        state_dict = module_dict.get(const.STATE_DICT)
        if state_dict:
            not_loaded_keys = module.load_state_dict(state_dict)
            logger.debug(f'Loaded encoder; extraneous keys: {not_loaded_keys}')

        return module

    def to_dict(self, include_state=True):
        module_dict = OrderedDict(
            {
                '__version__': kiwi.__version__,
                'class_name': self.__class__.__name__,
                const.CONFIG: json.loads(self.config.json()),
                const.STATE_DICT: self.state_dict() if include_state else None,
            }
        )
        return module_dict
