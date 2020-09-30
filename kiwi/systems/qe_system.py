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
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import more_itertools
import pytorch_lightning as pl
import torch
import torch.nn
from more_itertools import all_equal
from pydantic import PositiveInt, validator
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torchnlp.samplers import BucketBatchSampler

import kiwi
from kiwi import constants as const
from kiwi.data.batch import MultiFieldBatch
from kiwi.data.datasets.wmt_qe_dataset import WMTQEDataset
from kiwi.data.encoders.wmt_qe_data_encoder import WMTQEDataEncoder
from kiwi.systems._meta_module import MetaModule, Serializable
from kiwi.training import optimizers
from kiwi.utils.io import BaseConfig, convert_model_dict_if_needed, load_torch_file

logger = logging.getLogger(__name__)


class BatchSizeConfig(BaseConfig):
    train: PositiveInt = 1
    valid: PositiveInt = 1
    test: PositiveInt = 1


class ModelConfig(BaseConfig):
    encoder: Any = None
    decoder: Any = None
    outputs: Any = None
    tlm_outputs: Any = None


class QESystem(Serializable, pl.LightningModule, metaclass=ABCMeta):
    subclasses = {}

    class Config(BaseConfig):
        """System configuration base class."""

        class_name: Optional[str]
        """System class to use (must be a subclass of ``QESystem`` and decorated with
        ``@QESystem.register_subclass``)."""

        # Loadable configs
        load: Optional[Path]
        """Load pretrained Kiwi model.
        If set, system architecture and vocabulary parameters are ignored."""

        load_encoder: Optional[Path]
        """Load pretrained encoder (e.g., the Predictor).
        If set, encoder architecture and vocabulary parameters are ignored
        (for the fields that are part of the encoder)."""

        load_vocabs: Optional[Path]

        # This will be lazy loaded later, according to the selected `class_name` or
        # `load`
        model: Optional[Dict]
        """System specific options; they will be dynamically validated and instantiated
        depending of the ``class_name`` or ``load``."""

        data_processing: Optional[WMTQEDataEncoder.Config]

        optimizer: Optional[optimizers.OptimizerConfig]

        # Flags
        batch_size: BatchSizeConfig = 1
        num_data_workers: int = 4

        @validator('class_name', pre=True)
        def map_name_to_class(cls, v):
            if v in QESystem.subclasses:
                return v
            else:
                raise ValueError(
                    f'{v} is not a subclass of QESystem; make sure its class is '
                    f'decorated with `@QESystem.register_subclass`'
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

    def __init__(self, config, data_config: WMTQEDataset.Config = None):
        """Quality Estimation Base Class"""
        super().__init__()

        self.config = config

        self.data_config = data_config
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.data_encoders = None

        self._metrics = None
        self._main_metric_list = None
        self._main_metric_name = None
        self._main_metric_ordering = None

        # Module blocks
        self.encoder = None
        self.decoder = None
        self.outputs = None
        self.tlm_outputs = None

        # Load datasets
        if self.data_config:
            self.prepare_data()

    def _load_encoder(self, path: Path):
        logger.info(f'Loading encoder from {path}')
        module_dict = load_torch_file(path)
        module_dict = convert_model_dict_if_needed(module_dict)

        encoder_cls = MetaModule.retrieve_subclass(module_dict['encoder']['class_name'])
        self.data_encoders = WMTQEDataEncoder(
            config=self.config.data_processing,
            field_encoders=encoder_cls.input_data_encoders(self.config.model.encoder),
        )

        input_vocabs = {
            const.SOURCE: module_dict[const.VOCAB][const.SOURCE],
            const.TARGET: module_dict[const.VOCAB][const.TARGET],
        }
        if const.PE in module_dict[const.VOCAB]:
            input_vocabs[const.PE] = module_dict[const.VOCAB][const.PE]
        self.data_encoders.vocabularies_from_dict(input_vocabs, overwrite=True)

        self.encoder = MetaModule.from_dict(
            module_dict['encoder'],
            vocabs=self.data_encoders.vocabularies,
            pre_load_model=False,
        )

    def set_config_options(
        self,
        optimizer_config: optimizers.OptimizerConfig = None,
        batch_size: BatchSizeConfig = None,
        num_data_workers: int = None,
        data_config: WMTQEDataset.Config = None,
    ):
        if optimizer_config:
            self.config.optimizer = optimizer_config
        if batch_size:
            self.config.batch_size = batch_size
        if num_data_workers is not None:
            self.config.num_data_workers = num_data_workers
        if data_config:
            self.data_config = data_config
            self.prepare_data()

    # @property
    # def vocabs(self):
    #     return self.data_encoders.vocabularies

    # @property
    # def hparams(self):
    #     """This is a hack in order to have PyTorch-Lightning saving the config inside
    #     a checkpoint.
    #     """
    #     # This works better than self.config.dict() because the later doesn't convert
    #     # PosixPath to str.
    #     config_dict = json.loads(self.config.json())
    #     return SimpleNamespace(**config_dict)  # PTL is going to call `vars(hparams)`

    # @hparams.setter
    # def hparams(self, hparams):
    #     self._hparams = hparams

    def prepare_data(self):
        """Initialize the data sources the model will use to create the data loaders."""

        if not self.data_config:
            raise ValueError(
                'No configuration for data provided; pass it in the constructor or '
                'call `set_config_options(data_config=data_config)`'
            )
        # Initialize data reading
        if self.data_config.train:
            self.train_dataset, self.valid_dataset = WMTQEDataset.build(
                config=self.data_config, train=True, valid=True
            )
        elif self.data_config.valid:
            self.valid_dataset = WMTQEDataset.build(
                config=self.data_config, train=False, valid=True
            )
        if self.data_config.test:
            self.test_dataset = WMTQEDataset.build(config=self.data_config, test=True)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a PyTorch DataLoader for the training set.

        Requires calling ``prepare_data`` beforehand.

        Return:
            PyTorch DataLoader
        """
        sampler = BucketBatchSampler(
            RandomSampler(self.train_dataset),
            batch_size=self.config.batch_size.train,
            drop_last=False,
            sort_key=lambda sample: len(
                self.train_dataset[sample][const.TARGET].split()
            ),
            # bucket_size_multiplier=100,
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.config.num_data_workers,
            collate_fn=self.data_encoders.collate_fn,
            pin_memory=torch.cuda.is_initialized(),  # NOQA
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
            # sort_key=train_dataset.sort_key,
            # biggest_batches_first=True,
            # bucket_size_multiplier=model_options.__dict__.get('buffer_size'),
            # shuffle=True,
        )
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_sampler=sampler,
            num_workers=self.config.num_data_workers,
            collate_fn=self.data_encoders.collate_fn,
            pin_memory=torch.cuda.is_initialized(),  # NOQA
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.test_dataset:
            return None

        return self.prepare_dataloader(
            self.test_dataset,
            batch_size=self.config.batch_size.test,
            num_workers=self.config.num_data_workers,
        )

    def prepare_dataloader(
        self, dataset: WMTQEDataset, batch_size: int = 1, num_workers: int = 0
    ):
        sampler = BatchSampler(
            SequentialSampler(dataset), batch_size=batch_size, drop_last=False
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=self.data_encoders.collate_fn,
            pin_memory=torch.cuda.is_initialized(),  # NOQA
        )

    # def make_test_dataloader(
    #     self,
    #     samples: Dict[str, Iterable[str]],
    #     batch_size: int = 1,
    #     num_workers: int = 0,
    # ):
    #     dataset = WMTQEDataset(samples)
    #     return self.prepare_dataloader(dataset, batch_size, num_workers)

    def forward(self, batch_inputs):
        encoder_features = self.encoder(batch_inputs)
        features = self.decoder(encoder_features, batch_inputs)
        outputs = self.outputs(features, batch_inputs)

        # For fine-tuning the encoder
        if self.tlm_outputs:
            outputs.update(self.tlm_outputs(encoder_features, batch_inputs))

        return outputs

    def training_step(
        self, batch: MultiFieldBatch, batch_idx: int
    ) -> Dict[str, Dict[str, Tensor]]:
        model_out = self(batch)
        loss_dict = self.loss(model_out, batch)
        # avoid calling metrics when bs == 1 since it breaks due to dimensionality
        if batch['target'].tensor.size(0) != 1:
            metrics = self.metrics_step(batch, model_out, loss_dict)
            metrics_summary = self.metrics_end([metrics])
        else:
            metrics = {}
            metrics_summary = {self._main_metric_name: 0}
        return dict(
            loss=loss_dict[const.LOSS],
            metrics=metrics_summary,
            log=metrics_summary,
            progress_bar={
                self._main_metric_name: metrics_summary[self._main_metric_name],
            },  # optional (MUST ALL BE TENSORS)
        )

    def training_epoch_end(
        self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, Tensor]]]:
        # Compute macro averages of loss and metrics
        loss_avg = torch.tensor([out['loss'] for out in outputs]).mean()
        summary = defaultdict(lambda: torch.tensor(0.0))
        for output in outputs:
            for metric, value in output['metrics'].items():
                summary[metric] = summary[metric] + value
        for metric in summary:
            summary[metric] = summary[metric] / len(outputs)
        main_metric_dict = {self._main_metric_name: summary[self._main_metric_name]}
        return dict(
            loss=loss_avg,
            # metrics=summary,
            log=summary,
            progress_bar=main_metric_dict,
            # **main_metric_dict,
        )

    def validation_step(self, batch, batch_idx):
        model_out = self(batch)
        loss_dict = self.loss(model_out, batch)
        # avoid calling metrics when bs == 1 since it breaks due to dimensionality
        if batch['target'].tensor.size(0) != 1:
            metrics = self.metrics_step(batch, model_out, loss_dict)
        else:
            metrics = {}
        return dict(val_losses=loss_dict, val_metrics=metrics)

    def validation_epoch_end(
        self, outputs: List[Dict[str, Dict[str, Tensor]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        losses = defaultdict(lambda: torch.tensor(0.0))
        for output in outputs:
            for loss, value in output['val_losses'].items():
                losses[loss] = losses[loss] + value
        for loss in losses:
            losses[loss] = losses[loss] / len(outputs)

        val_loss = losses.pop(const.LOSS)
        summary = {f'val_loss_{loss}': value for loss, value in losses.items()}
        summary.update(
            self.metrics_end(
                [output['val_metrics'] for output in outputs], prefix='val_'
            )
        )
        metrics_message = textwrap.fill(
            ', '.join(['{}: {:0.4f}'.format(k, v) for k, v in summary.items()]),
            width=80,
            initial_indent='\t',
            subsequent_indent='\t',
        )
        logger.info(f'Validation metrics:\n{metrics_message}\n')
        main_metric_dict = {
            f'val_{self._main_metric_name}': summary[f'val_{self._main_metric_name}']
        }
        return dict(val_loss=val_loss, log=summary, progress_bar=main_metric_dict)

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        return super().test_step(*args, **kwargs)

    def test_epoch_end(
        self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        return super().test_epoch_end(outputs)

    def loss(self, model_out, batch):
        loss_dict = self.outputs.loss(model_out, batch)

        if self.tlm_outputs:
            extra_loss_dict = self.tlm_outputs.loss(model_out, batch)
            total_loss = loss_dict[const.LOSS] + extra_loss_dict[const.LOSS]
            loss_dict.update(extra_loss_dict)
            loss_dict[const.LOSS] = total_loss

        return loss_dict

    def metrics_step(self, batch, model_out, loss_dict):
        metrics_dict = self.outputs.metrics_step(batch, model_out, loss_dict)
        if self.tlm_outputs:
            metrics_dict.update(
                self.tlm_outputs.metrics_step(batch, model_out, loss_dict)
            )
        return metrics_dict

    def metrics_end(self, steps, prefix=''):
        metrics_dict = self.outputs.metrics_end(steps, prefix=prefix)
        if self.tlm_outputs:
            metrics_dict.update(self.tlm_outputs.metrics_end(steps, prefix=prefix))
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
        metric in the outputs. Subsequent calls return the configured main metric.
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
                names = [self.outputs.metrics[0].name]
                ordering = self.outputs.metrics[0].best_ordering
            else:
                metrics = {m.name: m for m in self.outputs.metrics}
                if isinstance(selected_metric, (list, tuple)):
                    selected = []
                    for selection in selected_metric:
                        if selection not in metrics:
                            raise KeyError(
                                f'Main metric {selection} is not a configured metric; '
                                f'available options are: {list(metrics.keys())}'
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
    def from_config(config: Config, data_config: WMTQEDataset.Config = None):
        if config.load:
            system = QESystem.load(config.load)
            system.set_config_options(
                optimizer_config=config.optimizer,
                batch_size=config.batch_size,
                data_config=data_config,
            )
        else:
            system_cls = QESystem.retrieve_subclass(config.class_name)
            # Re-instantiate the config object in order to get the proper ModelConfig
            # validated and converted.
            config = system_cls.Config(**config.dict())
            system = system_cls(config=config, data_config=data_config)

        return system

    @classmethod
    def load(cls, path: Path, map_location=None):
        logger.info(f'Loading system from {path}')
        module_dict = load_torch_file(path, map_location=map_location)
        system = QESystem.from_dict(module_dict=module_dict)
        return system

    @classmethod
    def from_dict(cls, module_dict: Dict[str, Any]):
        module_dict = convert_model_dict_if_needed(module_dict)
        system_cls = cls.retrieve_subclass(module_dict['class_name'])
        config = system_cls.Config(**module_dict[const.CONFIG])
        system = system_cls(config=config, module_dict=module_dict)
        # The constructor is now responsible for calling `_load_dict()`
        return system

    def _load_dict(self, module_dict):
        encoder_cls = MetaModule.retrieve_subclass(module_dict['encoder']['class_name'])
        self.data_encoders = WMTQEDataEncoder(
            config=self.config.data_processing,
            field_encoders=encoder_cls.input_data_encoders(self.config.model.encoder),
        )
        vocabs = self.data_encoders.vocabularies_from_dict(module_dict[const.VOCAB])

        self.encoder = MetaModule.from_dict(
            module_dict=module_dict['encoder'], vocabs=vocabs, pre_load_model=False
        )
        self.decoder = MetaModule.from_dict(
            module_dict=module_dict['decoder'], inputs_dims=self.encoder.size()
        )
        self.outputs = MetaModule.from_dict(
            module_dict=module_dict['outputs'],
            inputs_dims=self.decoder.size(),
            vocabs=vocabs,
        )
        if module_dict['tlm_outputs'] is not None:
            self.tlm_outputs = MetaModule.from_dict(
                module_dict=module_dict['tlm_outputs'],
                inputs_dims=self.encoder.size(),
                vocabs=vocabs,
            )

    def to_dict(self, include_state=True):
        # TODO: add consts
        model_dict = OrderedDict(
            {
                '__version__': kiwi.__version__,
                'class_name': self.__class__.__name__,
                'config': json.loads(self.config.json()),  # Round-trip to remove nests
                'vocab': self.data_encoders.vocabularies,
                # 'data_encoders': self.data_encoders.to_dict(),
                'encoder': self.encoder.to_dict(),
                'decoder': self.decoder.to_dict(),
                'outputs': self.outputs.to_dict(),
                'tlm_outputs': self.tlm_outputs.to_dict() if self.tlm_outputs else None,
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

        Return: for compatibility with PyTorch-Lightning, any of these 3 options:
            - Single optimizer
            - List or Tuple - List of optimizers
            - Tuple of Two lists - The first with multiple optimizers, the second with
                                   learning-rate schedulers
        """
        hidden_size = getattr(self.config.model.encoder, 'hidden_size', None)
        if self.config.optimizer.encoder_learning_rate:
            parameters = [
                {"params": self.outputs.parameters()},
                {"params": self.decoder.parameters()},
                {
                    "params": self.encoder.parameters(),
                    "lr": self.config.optimizer.encoder_learning_rate,
                },
            ]
            if self.tlm_outputs:
                parameters.append({"params": self.tlm_outputs.parameters()})
        else:
            parameters = self.parameters()

        return optimizers.from_config(
            self.config.optimizer, parameters, model_size=hidden_size
        )

    def predict(self, batch_inputs, positive_class_label=const.BAD):
        model_out = self(batch_inputs)

        predictions = self.outputs.decode_outputs(
            model_out, batch_inputs, positive_class_label
        )

        if const.TARGET_TAGS in predictions and const.GAP_TAGS in predictions:
            targetgaps = []
            for target, gaps in zip(
                predictions[const.TARGET_TAGS], predictions[const.GAP_TAGS]
            ):
                # Order is important (gaps, then target)
                targetgaps.append(list(more_itertools.roundrobin(gaps, target)))
            predictions[const.TARGETGAPS_TAGS] = targetgaps

            targetgaps_labels = []
            for target, gaps in zip(
                predictions[f'{const.TARGET_TAGS}_labels'],
                predictions[f'{const.GAP_TAGS}_labels'],
            ):
                # Order is important (gaps, then target)
                targetgaps_labels.append(list(more_itertools.roundrobin(gaps, target)))
            predictions[f'{const.TARGETGAPS_TAGS}_labels'] = targetgaps_labels

        return predictions
