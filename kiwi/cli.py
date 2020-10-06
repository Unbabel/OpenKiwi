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
"""
Kiwi runner
~~~~~~~~~~~

Quality Estimation toolkit.

Invoke as ``kiwi PIPELINE``.

Usage:
    kiwi [options] (train|pretrain|predict|evaluate) CONFIG_FILE [OVERWRITES ...]
    kiwi (-h | --help | --version | --example)

Pipelines:
    train          Train a QE model
    pretrain       Pretrain a TLM model to be used as an encoder for a QE model
    predict        Use a pre-trained model for prediction
    evaluate       Evaluate a model's predictions using popular metrics
    search         Search training hyper-parameters for a QE model

Disabled pipelines:
    jackknife      Jackknife training data with model

Arguments:
    CONFIG_FILE    configuration file to use (e.g., config/nuqe.yaml)
    OVERWRITES     key=value to overwrite values in CONFIG_FILE; use ``key.subkey``
                   for nested keys.

Options:
    -v --verbose          log debug messages
    -q --quiet            log only warning and error messages
    -h --help             show this help message and exit
    --version             show version and exit
    --example             print an example configuration file

"""
import os.path
import sys
from pathlib import Path
from typing import Dict

import hydra.experimental
import hydra.utils
from docopt import docopt
from hydra._internal.hydra import Hydra
from omegaconf import OmegaConf

from kiwi import __version__
from kiwi.lib import evaluate, predict, pretrain, search, train


def arguments_to_configuration(arguments: Dict) -> Dict:
    config_file = Path(arguments['CONFIG_FILE'])

    # # Using OmegaConf
    # config_dict = load_config(config_file)
    # # Update config from command line arguments
    # config = OmegaConf.create(config_dict)
    # config.merge_with_dotlist(arguments.get('OVERWRITES', []))

    # Using Hydra
    relative_dir = Path(
        os.path.relpath(config_file.resolve().parent, start=Path(__file__).parent)
    )
    Hydra.create_main_hydra_file_or_module(
        calling_file=__file__,
        calling_module=None,
        config_dir=str(relative_dir),
        strict=False,
    )
    config = hydra.experimental.compose(
        config_file=config_file.name, overrides=arguments.get('OVERWRITES', [])
    )
    # print(config.pretty())

    # Back to a dictionary
    config_dict = OmegaConf.to_container(config)

    # config_dict['anchor_directory'] = config_file.parent
    config_dict['verbose'] = arguments.get('--verbose')
    config_dict['quiet'] = arguments.get('--quiet')

    return config_dict


def cli():
    arguments = docopt(
        __doc__, argv=sys.argv[1:], help=True, version=__version__, options_first=False
    )
    config_dict = arguments_to_configuration(arguments)

    if arguments['train']:
        train.train_from_configuration(config_dict)
    if arguments['predict']:
        predict.predict_from_configuration(config_dict)
    if arguments['pretrain']:
        pretrain.pretrain_from_configuration(config_dict)
    if arguments['evaluate']:
        evaluate.evaluate_from_configuration(config_dict)
    if arguments['search']:
        search.search_from_configuration(config_dict)
    # Meta Pipelines
    # if options.pipeline == 'jackknife':
    #     jackknife.main(extra_args)


if __name__ == '__main__':  # pragma: no cover
    cli()  # pragma: no cover
