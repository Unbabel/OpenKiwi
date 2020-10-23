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
    kiwi [options] (train|pretrain|predict|evaluate|search) CONFIG_FILE [OVERWRITES ...]
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
import sys

from docopt import docopt

from kiwi import __version__
from kiwi.lib import evaluate, predict, pretrain, search, train
from kiwi.lib.utils import arguments_to_configuration


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
