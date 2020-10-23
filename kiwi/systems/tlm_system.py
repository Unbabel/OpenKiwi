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
import textwrap
from abc import ABCMeta
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn
from more_itertools import all_equal
from pydantic import PositiveInt, validator
from torch.optim.optimizer import Optimizer
from torch.utils.data import BatchSampler, SequentialSampler

import kiwi
from kiwi import constants as const
from kiwi.data.datasets.parallel_dataset import ParallelDataset
from kiwi.data.encoders.parallel_data_encoder import ParallelDataEncoder
from kiwi.systems._meta_module import MetaModule, Serializable
from kiwi.training import optimizers
from kiwi.utils.io import BaseConfig, load_torch_file

logger = logging.getLogger(__name__)


class BatchSizeConfig(BaseConfig):
    train: PositiveInt = None
    valid: PositiveInt = None


class ModelConfig(BaseConfig):
    encoder: Any = None
    tlm_outputs: Any = None


class TLMSystem(Serializable, pl.LightningModule, metaclass=ABCMeta):
    subclasses = {}

    class Config(BaseConfig):
        """System configuration base class."""

        class_name: Optional[str]

        # Loadable configs
        load: Optional[Path]
        """If set, system architecture and vocabulary parameters are ignored.
        Load pretrained kiwi encoder model."""

        load_vocabs: Optional[Path]

        # This will be lazy loaded later, according to the selected `class_name` or
        # `load`
        model: Optional[Dict]

        data_processing: Optional[ParallelDataEncoder.Config]

        optimizer: optimizers.OptimizerConfig

        # Flags
        batch_size: BatchSizeConfig = 1
        num_data_workers: int = 4

        @validator('class_name', pre=True)
        def map_name_to_class(cls, v):
            if v in TLMSystem.subclasses:
                return v
            else:
                raise ValueError(
                    f'{v} is not a subclass of TLMSystem; make sure its class is '
                    f'decorated with `@TLMSystem.register_subclass`'
                )

        @validator('load', always=True)
        def check_consistency(cls, v, values):
            if v is None and values.get('class_name') is None:
                raise ValueError('Must provide `class_name` or `load`')
            if v is not None and values['class_name'] is not None:
                model_dict = load_torch_file(v)
                if model_dict['class_name'] != values['class_name']:
                    raise ValueError(
                        f'`class_name` in configuration file ({values["class_name"]}) '
                        f'does not match class_name in the loaded model file '
                        f'({model_dict["class_name"]}); consider removing `class_name`'
                    )
            return v

        @validator('model', pre=True, always=True)
        def check_model_requirement(cls, v, values):
            if v is None and not values.get('load'):
                raise ValueError('field required when not loading model')
            return v

        @validator('batch_size', pre=True, always=True)
        def check_batching(cls, v):
            if isinstance(v, int):
                return {'train': v, 'valid': v, 'test': v}
            return v

    def __init__(self, config, data_config: ParallelDataset.Config = None):
        """Translation Language Model Base Class."""
        super().__init__()

        self.config = config

        self.data_config = data_config
        self.train_dataset = None
        self.valid_dataset = None

        self.data_encoders = None

        self._metrics = None
        self._main_metric_list = None
        self._main_metric_name = None
        self._main_metric_ordering = None

        # Module blocks
        self.encoder = None
        self.tlm_outputs = None

        # Load datasets
        if self.data_config:
            self.prepare_data()

    def set_config_options(
        self,
        optimizer_config: optimizers.OptimizerConfig = None,
        batch_size: BatchSizeConfig = None,
        data_config: ParallelDataset.Config = None,
    ):
        if optimizer_config:
            self.config.optimizer = optimizer_config
        if batch_size:
            self.config.batch_size = batch_size
        if data_config:
            self.data_config = data_config
            self.prepare_data()

    def prepare_data(self):
        """Initialize the data sources that model will use to create the data loaders"""
        if not self.data_config:
            raise ValueError(
                'No configuration for data provided; pass it in the constructor or '
                'call `set_config_options(data_config=data_config)`'
            )
        self.train_dataset, self.valid_dataset = ParallelDataset.build(
            config=self.data_config, train=True, valid=True
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a PyTorch DataLoader for the training set.

        Requires calling ``prepare_data`` beforehand.

        Return:
            PyTorch DataLoader
        """
        sampler = BatchSampler(
            SequentialSampler(self.train_dataset),
            # RandomSampler(self.train_dataset),
            batch_size=self.config.batch_size.train,
            drop_last=False,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.config.num_data_workers,
            collate_fn=self.data_encoders.collate_fn,
            pin_memory=torch.cuda.is_initialized(),  # NoQA
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a PyTorch DataLoader for the validation set.

        Requires calling ``prepare_data`` beforehand.

        Return:
            PyTorch DataLoader
        """
        sampler = BatchSampler(
            SequentialSampler(self.valid_dataset),
            batch_size=self.config.batch_size.valid,
            drop_last=False,
        )
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_sampler=sampler,
            num_workers=self.config.num_data_workers,
            collate_fn=self.data_encoders.collate_fn,
            pin_memory=torch.cuda.is_initialized(),  # NoQA
        )

    def forward(self, batch_inputs) -> Dict:
        """Same as `torch.nn.Module.forward()`.

        In Kiwi we use it to glue together the modular parts that constitute a model,
        e.g., the encoder and a tlm_output.

        Arguments:
            batch_inputs: Dict containing a batch of data. See
                `kiwi.data.encoders.field_encoders.QEEncoder.batch_encode()`.

        Return:
            outputs: outputs of the tlm_outputs module.
        """
        encoder_features = self.encoder(batch_inputs)
        outputs = self.tlm_outputs(encoder_features, batch_inputs)

        return outputs

    def training_step(self, batch, batch_idx):
        model_out = self(batch)
        loss_dict = self.loss(model_out, batch)
        metrics = self.metrics_step(batch, model_out, loss_dict)
        metrics_summary = self.metrics_end([metrics])
        return dict(
            loss=loss_dict[const.LOSS],
            metrics=metrics_summary,
            log=metrics_summary,
            progress_bar={
                self._main_metric_name: metrics_summary[self._main_metric_name],
            },  # optional (MUST ALL BE TENSORS)
        )

    def validation_step(self, batch, batch_idx):
        model_out = self(batch)
        loss_dict = self.loss(model_out, batch)
        metrics = self.metrics_step(batch, model_out, loss_dict)
        return dict(val_loss=loss_dict[const.LOSS], val_metrics=metrics)

    def validation_epoch_end(self, outputs: list):
        loss_avg = torch.tensor([out['val_loss'] for out in outputs]).mean()
        summary = {'val_loss': loss_avg}
        summary.update(
            self.metrics_end(
                [output['val_metrics'] for output in outputs], prefix='val_'
            )
        )
        metrics_message = textwrap.fill(
            ', '.join(['{}: {:0.4f}'.format(k, v) for k, v in summary.items()]),
            width=70,
            subsequent_indent='\t',
        )
        logger.info(f'Validation metrics:\n' f'{metrics_message}\n')
        main_metric_dict = {
            f'val_{self._main_metric_name}': summary[f'val_{self._main_metric_name}']
        }
        return dict(val_loss=loss_avg, log=summary, progress_bar=main_metric_dict)

    def loss(self, model_out, batch) -> Dict:
        """Compute total model loss.

        Return:
            loss_dict: Dict[loss_key]=value
        """
        loss_dict = self.tlm_outputs.loss(model_out, batch)

        return loss_dict

    def metrics_step(self, batch, model_out, loss_dict):
        metrics_dict = self.tlm_outputs.metrics_step(batch, model_out, loss_dict)
        return metrics_dict

    def metrics_end(self, steps, prefix=''):
        metrics_dict = self.tlm_outputs.metrics_end(steps, prefix=prefix)
        if len(self._main_metric_list) > 1:
            metrics_dict[f'{prefix}{self._main_metric_name}'] = sum(
                metrics_dict[f'{prefix}{metric}'] for metric in self._main_metric_list
            )
        return metrics_dict

    def main_metric(
        self, selected_metric: Union[str, List[str]] = None
    ) -> (Union[str, List[str]], str):
        """Configure and retrieve the metric to be used for monitoring.

        The first time it is called, the main metric is configured based on the
        specified metrics in ``selected_metric`` or, if not provided, on the first
        metric in the TLM outputs. Subsequent calls return the configured main metric.
        If a subsequent call specifies ``selected_metric``, configuration is done again.

        Return:
             a tuple containing the main metric name and the ordering.
                 Note that the first element might be a concatenation of several
                 metrics in case ``selected_metric`` is a list. This is useful for
                 considering more than one metric as the best
                 (``metric_end()`` will sum over them).
        """
        if self._main_metric_list is None or selected_metric is not None:
            if not selected_metric:
                names = [self.tlm_outputs.metrics[0].name]
                ordering = self.tlm_outputs.metrics[0].best_ordering
            else:
                metrics = {m.name: m for m in self.tlm_outputs.metrics}
                if isinstance(selected_metric, (list, tuple)):
                    selected = []
                    for selection in selected_metric:
                        if selection not in metrics:
                            raise KeyError(
                                f'Main metric {selection} is not a configured metric; '
                                f'available options are: {metrics.keys()}'
                            )
                        selected.append(metrics[selection])
                    names = [m.name for m in selected]
                    orderings = [m.best_ordering for m in selected]
                    if not all_equal(orderings):
                        raise ValueError(
                            f'Main metrics {names} have different '
                            f'ordering: {orderings}'
                        )
                    ordering = orderings[0]
                else:
                    try:
                        selected = metrics[selected_metric]
                    except KeyError:
                        raise KeyError(
                            f'Main metric {selected_metric} is not a configured metric;'
                            f' available options are: {metrics.keys()}'
                        )
                    names = [selected.name]
                    ordering = selected.best_ordering
            self._main_metric_list = names
            self._main_metric_name = '+'.join(names)
            self._main_metric_ordering = ordering
        return self._main_metric_name, self._main_metric_ordering

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def from_config(config: Config, data_config: ParallelDataset.Config = None):
        if config.load:
            system = TLMSystem.load(config.load)
            system.set_config_options(
                optimizer_config=config.optimizer,
                batch_size=config.batch_size,
                data_config=data_config,
            )
        else:
            system_cls = TLMSystem.retrieve_subclass(config.class_name)
            # Re-instantiate the config object in order to get the proper ModelConfig
            # validated and converted.
            config = system_cls.Config(**config.dict())
            system = system_cls(config=config, data_config=data_config)

        return system

    @classmethod
    def load(cls, path: Path, map_location=None):
        logger.info(f'Loading system from {path}')
        module_dict = load_torch_file(path, map_location=map_location)
        system = TLMSystem.from_dict(module_dict=module_dict)
        return system

    @classmethod
    def from_dict(cls, module_dict: Dict[str, Any]):
        system_cls = cls.retrieve_subclass(module_dict['class_name'])
        config = system_cls.Config(**module_dict[const.CONFIG])
        system = system_cls(config=config, module_dict=module_dict)
        # The constructor is now responsible for calling `_load_dict()`
        return system

    def _load_dict(self, module_dict):
        encoder_cls = MetaModule.retrieve_subclass(module_dict['encoder']['class_name'])
        self.data_encoders = ParallelDataEncoder(
            config=self.config.data_processing,
            field_encoders=encoder_cls.input_data_encoders(self.config.model.encoder),
        )
        vocabs = self.data_encoders.vocabularies_from_dict(module_dict[const.VOCAB])

        self.encoder = MetaModule.from_dict(
            module_dict=module_dict['encoder'], vocabs=vocabs, pretraining=True
        )
        self.tlm_outputs = MetaModule.from_dict(
            module_dict=module_dict['tlm_outputs'],
            inputs_dims=self.encoder.size(),
            vocabs=vocabs,
            pretraining=True,
        )

    def to_dict(self, include_state=True):
        # TODO: add consts
        model_dict = OrderedDict(
            {
                '__version__': kiwi.__version__,
                'class_name': self.__class__.__name__,
                'config': json.loads(self.config.json()),
                'vocab': self.data_encoders.vocabularies,
                'encoder': self.encoder.to_dict(),
                'tlm_outputs': self.tlm_outputs.to_dict(),
            }
        )
        return model_dict

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(self.to_dict())
        # Clean up PTL mess
        checkpoint['state_dict'] = {}
        checkpoint['hparams'] = {}

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Optional[
            Union[Dict[str, str], str, torch.device, int, Callable]
        ] = None,
        tags_csv: Optional[str] = None,
        *args,
        **kwargs,
    ) -> 'pl.LightningModule':
        return cls.load(path=Path(checkpoint_path), map_location=map_location)

    def configure_optimizers(
        self,
    ) -> Optional[Union[Optimizer, Sequence[Optimizer], Tuple[List, List]]]:
        """Instantiate configured optimizer and LR scheduler.

        Returns: for compatibility with PyTorch-Lightning, any of these 3 options:
            - Single optimizer
            - List or Tuple - List of optimizers
            - Tuple of Two lists - The first with multiple optimizers, the second with
                                   learning-rate schedulers
        """
        hidden_size = getattr(self.config.model.encoder, 'hidden_size', None)
        return optimizers.from_config(
            self.config.optimizer, self.parameters(), model_size=hidden_size
        )
