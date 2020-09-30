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

from kiwi.lib.train import Configuration as TrainConfig
from kiwi.lib.train import TrainRunInfo, run
from kiwi.lib.utils import load_config
from kiwi.systems.tlm_system import TLMSystem

logger = logging.getLogger(__name__)


class Configuration(TrainConfig):
    system: TLMSystem.Config


def pretrain_from_file(filename) -> TrainRunInfo:
    """Load options from a config file and call the pretraining procedure.

    Arguments:
        filename: of the configuration file.

    Return:
        object with training information.
    """
    config = load_config(filename)
    return pretrain_from_configuration(config)


def pretrain_from_configuration(configuration_dict) -> TrainRunInfo:
    """Run the entire training pipeline using the configuration options received.

    Arguments:
        configuration_dict: dictionary with config options.

    Return:
        object with training information.
    """
    config = Configuration(**configuration_dict)
    train_info = run(config, TLMSystem)
    return train_info
