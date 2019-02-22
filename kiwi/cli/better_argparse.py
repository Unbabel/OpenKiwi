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

import configargparse
from configargparse import Namespace

from kiwi.cli import opts
from kiwi.cli.opts import PathType
from kiwi.lib.utils import merge_namespaces

logger = logging.getLogger(__name__)


class HyperPipelineParser:
    def __init__(
        self, name, pipeline_parser, pipeline_config_key, options_fn=None
    ):
        self.name = name
        self._pipeline_parser = pipeline_parser
        self._pipeline_config_key = pipeline_config_key.replace('-', '_')

        self._parser = configargparse.get_argument_parser(
            self.name,
            prog='kiwi {}'.format(self.name),
            add_help=False,
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            ignore_unknown_config_file_keys=False,
        )

        self._parser.add(
            '--config',
            required=False,
            is_config_file=True,
            type=PathType(exists=True),
            help='Load config file from path',
        )

        if options_fn is not None:
            options_fn(self._parser)

    def parse(self, args):
        if len(args) == 1 and args[0] in ['-h', '--help']:
            self._parser.print_help()
            return None

        # Parse train pipeline options
        meta_options, extra_args = self._parser.parse_known_args(args)
        print(meta_options)

        if hasattr(meta_options, self._pipeline_config_key):
            extra_args = [
                '--config',
                getattr(meta_options, self._pipeline_config_key),
            ] + extra_args

        pipeline_options = self._pipeline_parser.parse(extra_args)

        options = Namespace()
        options.meta = meta_options
        options.pipeline = pipeline_options

        return options


class PipelineParser:
    _parsers = {}

    def __init__(
        self,
        name,
        model_parsers,
        options_fn=None,
        add_io_options=True,
        add_general_options=True,
        add_logging_options=True,
        add_save_load_options=True,
    ):
        self.name = name

        # Give the option to create pipelines with no models
        if model_parsers is not None:
            self._models = {model.name: model for model in model_parsers}
        else:
            self._models = None

        if name in self._parsers:
            self._parser = self._parsers[name]
        else:
            self._parser = configargparse.get_argument_parser(
                self.name,
                add_help=False,
                prog='kiwi {}'.format(self.name),
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                ignore_unknown_config_file_keys=True,
            )
            self._parsers[name] = self._parser
            self.add_config_option(self._parser)

            if add_io_options:
                opts.io_opts(self._parser)
            if add_general_options:
                opts.general_opts(self._parser)
            if add_logging_options:
                opts.logging_opts(self._parser)
            if add_save_load_options:
                opts.save_load_opts(self._parser)

            if options_fn is not None:
                options_fn(self._parser)

            if model_parsers is not None:
                group = self._parser.add_argument_group('models')
                group.add_argument(
                    '--model',
                    required=True,
                    choices=self._models.keys(),
                    help="Use 'kiwi {} --model <model> --help' for specific "
                    "model options.".format(self.name),
                )

        if 'config' in self._parsers:
            self._config_option_parser = self._parsers['config']
        else:
            self._config_option_parser = configargparse.get_argument_parser(
                'config', add_help=False
            )
            self._parsers['config'] = self._config_option_parser
            self.add_config_option(self._config_option_parser, read_file=False)

    @staticmethod
    def add_config_option(parser, read_file=True):
        parser.add(
            '--config',
            required=False,
            is_config_file=read_file,
            help='Load config file from path',
        )

    def parse_config_file(self, file_name):
        return self.parse(['--config', str(file_name)])

    def parse(self, args):
        if len(args) == 1 and args[0] in ['-h', '--help']:
            self._parser.print_help()
            return None

        # Parse train pipeline options
        pipeline_options, extra_args = self._parser.parse_known_args(args)
        config_option, _ = self._config_option_parser.parse_known_args(args)

        options = Namespace()
        options.pipeline = pipeline_options
        options.model = None
        options.model_api = None

        # Parse specific model options if there are model parsers
        if self._models is not None:
            if pipeline_options.model not in self._models:
                raise KeyError(
                    'Invalid model: {}'.format(pipeline_options.model)
                )

            if config_option:
                extra_args = ['--config', config_option.config] + extra_args

            # Check if there are model parsers
            model_parser = self._models[pipeline_options.model]
            model_options, remaining_args = model_parser.parse_known_args(
                extra_args
            )

            options.model = model_options
            # Retrieve the respective API for the selected model
            options.model_api = model_parser.api_module
        else:
            remaining_args = extra_args

        options.all_options = merge_namespaces(options.pipeline, options.model)

        if remaining_args:
            raise KeyError('Unrecognized options: {}'.format(remaining_args))

        return options


class ModelParser:
    _parsers = {}

    def __init__(self, name, pipeline, options_fn, api_module, title=None):
        self.name = name
        self._title = title
        self._pipeline = pipeline
        self.api_module = api_module

        self._parser = self.get_parser(
            '{}-{}'.format(name, pipeline), description=self._title
        )
        PipelineParser.add_config_option(self._parser)
        options_fn(self._parser)

    @classmethod
    def get_parser(cls, name, **kwargs):
        if name in cls._parsers:
            return cls._parsers[name]
        parser = configargparse.get_argument_parser(
            name,
            prog='... {}'.format(name),
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            ignore_unknown_config_file_keys=True,
            **kwargs,
        )
        cls._parsers[name] = parser
        return parser

    def parse_known_args(self, args):
        return self._parser.parse_known_args(args)
