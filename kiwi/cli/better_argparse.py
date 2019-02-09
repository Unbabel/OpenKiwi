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

        # self._config_option_parser = configargparse.get_argument_parser(
        #     name='meta-config', add_help=False)

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
        # options.all_options = merge_namespaces(meta_options,
        #                                        pipeline_options)

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
        add_save_load_options=True,
    ):
        self.name = name

        self._models = {model.name: model for model in model_parsers}

        self._parser = self.get_parser(
            self.name,
            prog='kiwi {}'.format(self.name),
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            ignore_unknown_config_file_keys=True,
        )

        self._config_option_parser = self.get_parser(name='config')

        self.add_config_option(self._parser)
        self.add_config_option(self._config_option_parser, read_file=False)

        if add_io_options:
            opts.io_opts(self._parser)
        if add_general_options:
            opts.general_opts(self._parser)
        if add_save_load_options:
            opts.save_load_opts(self._parser)

        if options_fn is not None:
            options_fn(self._parser)

        group = self._parser.add_argument_group('models')
        group.add_argument(
            '--model',
            required=True,
            choices=self._models.keys(),
            help="Use 'kiwi {} --model <model> --help' for specific "
            "model options.".format(self.name),
        )

    @classmethod
    def get_parser(cls, name, add_help=False, **kwargs):
        if name in cls._parsers:
            return cls._parsers[name]
        parser = configargparse.get_argument_parser(
            name, add_help=add_help, **kwargs
        )
        cls._parsers[name] = parser
        return parser

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
        # Parse specific model options
        if pipeline_options.model not in self._models:
            raise KeyError('Invalid model: {}'.format(pipeline_options.model))

        config_option, _ = self._config_option_parser.parse_known_args(args)
        if config_option:
            extra_args = ['--config', config_option.config] + extra_args

        model_parser = self._models[pipeline_options.model]
        model_options, remaining_args = model_parser.parse_known_args(
            extra_args
        )

        if remaining_args:
            raise KeyError('Unrecognized options: {}'.format(remaining_args))

        options = Namespace()
        options.pipeline = pipeline_options
        options.model = model_options
        options.all_options = merge_namespaces(pipeline_options, model_options)
        # Retrieve the respective API for the selected model
        options.model_api = model_parser.api_module

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
            **kwargs
        )
        cls._parsers[name] = parser
        return parser

    def parse_known_args(self, args):
        return self._parser.parse_known_args(args)
