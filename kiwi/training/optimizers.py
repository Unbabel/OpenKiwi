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
import math
from pathlib import Path
from statistics import mode
from typing import Any, Iterator, List, Tuple, Union

import torch
import torch.optim
from pydantic import validator
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from transformers import AdamW, get_linear_schedule_with_warmup

from kiwi.utils.io import BaseConfig

logger = logging.getLogger(__name__)


class OptimizerConfig(BaseConfig):
    class_name: str
    learning_rate: float
    """Starting learning rate. Recommended settings:
    sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001"""

    encoder_learning_rate: float = None
    """Different learning rate for the encoder. If set, the encoder will use a different
    learning rate from the rest of the parameters."""

    warmup_steps: Union[float, int] = None
    """Increase the learning rate until X steps. Integer steps for `noam` optimizer and
    `adamw`. If float, use it as portion of ``training_steps``."""

    training_steps: int = None
    """Total number of training steps. Used for the `adamw` optimizer. If not specified,
    use training dataset size."""

    learning_rate_decay: float = 1.0
    """Factor by which the learning rate will be reduced: ``new_lr = lr * factor``.
    Scheduler is only used if this is greater than 0."""

    learning_rate_decay_start: int = 2
    """Number of epochs with no improvement after which learning rate will be reduced.
    Only applicable if ``learning_rate_decay`` is greater than 0."""

    load: Path = None

    @validator('warmup_steps', pre=False)
    def cast_steps(cls, v):
        if v is not None:
            if 0 <= v <= 1:
                return float(v)
            elif float(v) == int(v):
                return int(v)
            else:
                return float(v)
        return v


def get_noam_decay_schedule(
    optimizer: Optimizer, num_warmup_steps: int, model_size: int
):
    """Create a schedule with the learning rate decay strategy from the AIAYN paper.

    Arguments:
        optimizer: wrapped optimizer.
        num_warmup_steps: the number of steps to linearly increase the learning rate.
        model_size: the hidden size parameter which dominates the number of
                    parameters in your model.
    """

    def lr_lambda(current_step: int) -> float:
        """Compute a multiplicative factor given an integer parameter epoch."""
        current_step += 1
        sqrt_model_size = math.pow(float(model_size), -0.5)
        sqrt_warmup_steps = math.pow(float(num_warmup_steps), -1.5)
        sqrt_step = math.pow(float(current_step), -0.5)
        return sqrt_model_size * min(sqrt_step, float(current_step) * sqrt_warmup_steps)

    return LambdaLR(optimizer, lr_lambda)


OPTIMIZERS_MAPPING = {
    'sgd': torch.optim.SGD,
    'averaged_sgd': torch.optim.ASGD,
    'adam': torch.optim.Adam,
    'noam': torch.optim.Adam,  # Noam is actually a scheduler
    'sparse_adam': torch.optim.SparseAdam,
    'adamw': AdamW,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'adamax': torch.optim.Adamax,
    'rmsprop': torch.optim.RMSprop,
}


def optimizer_class(name):
    if name not in OPTIMIZERS_MAPPING:
        raise RuntimeError(f'Invalid optimizer: {name}')
    return OPTIMIZERS_MAPPING[name]


def optimizer_name(cls):
    for name, klass in OPTIMIZERS_MAPPING.items():
        if klass == cls:
            return name
    raise RuntimeError(f'Invalid optimizer class: {cls}')


def from_config(
    config: OptimizerConfig,
    parameters: Iterator[Parameter],
    model_size: int = None,
    training_data_size: int = None,
) -> Union[
    Optimizer, List[Optimizer], Tuple[List[Optimizer], List[Any]],
]:
    """

    Arguments:
        config: common options shared by most optimizers
        parameters: model parameters
        model_size: required for the Noam LR schedule; if not provided, the mode of all
                    parameters' last dimension is used

    Return: for compatibility with PyTorch-Lightning, any of these 3 options:
        - Single optimizer
        - List or Tuple - List of optimizers
        - Tuple of Two lists - The first with multiple optimizers, the second with
                               learning-rate schedulers

    Notes:
        We currently never return multiple optimizers or schedulers, so option 2 above
        is not taking place yet. Option 3 returns a single optimizer and scheduler
        inside lists.
    """

    if config.load:
        optimizer_dict = torch.load(
            config.load, map_location=lambda storage, loc: storage
        )
        optimizer = optimizer_class(optimizer_dict['name'])(
            filter(lambda p: p.requires_grad, parameters), lr=config.learning_rate,
        )
        optimizer.load_state_dict(optimizer_dict['state_dict'])
    else:
        optimizer_cls = optimizer_class(config.class_name)
        opt_kwargs = {}
        if config.class_name == 'noam':
            opt_kwargs = {
                'betas': (0.9, 0.98),  # in AIAYN they use (0.9, 0.98)
                'eps': 1e-9,
            }
        elif config.class_name == 'adamw':
            # To reproduce BertAdam specific behavior set correct_bias=False
            opt_kwargs = {'correct_bias': False}
        # TODO: make this more elegant
        if config.encoder_learning_rate:
            for el in parameters:
                el['params'] = filter(lambda p: p.requires_grad, el['params'])
        else:
            parameters = filter(lambda p: p.requires_grad, parameters)
        optimizer = optimizer_cls(parameters, lr=config.learning_rate, **opt_kwargs,)
    logger.info(str(optimizer))

    scheduler = None
    if config.class_name == 'adamw':
        if not config.training_steps:
            if not training_data_size:
                raise ValueError(
                    'AdamW optimizer needs to have `training_steps` configured or'
                    'training data size must be provided.'
                )
            logger.info(
                f'Optimizer training steps not set; using training data size: '
                f'{training_data_size}'
            )
            config.training_steps = training_data_size
        if isinstance(config.warmup_steps, float):
            fraction_steps = config.warmup_steps
            config.warmup_steps = config.training_steps * fraction_steps
            logger.info(
                f'Optimizer warm-up steps fraction ({fraction_steps}) converted to '
                f'{config.warmup_steps} steps'
            )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.training_steps,
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }
    elif config.class_name == 'noam':
        if model_size is None:
            # Use mode of last dimensions from all parameters.
            model_size = mode([p.size(-1) for p in parameters])
            logger.info(
                'Using Noam optimizer requires knowing the dominating '
                'dimensionality; no ``model_size`` argument was provided, so we '
                'will use the mode of the last dimension of all model parameters: '
                f'{model_size}.'
            )
        scheduler = get_noam_decay_schedule(
            optimizer, max(0, int(config.warmup_steps)), model_size
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch',
        }
    elif 0.0 < config.learning_rate_decay < 1.0:
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=config.learning_rate_decay,
            patience=config.learning_rate_decay_start,
            verbose=True,
            mode='max',
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch',
        }

    if scheduler is not None:
        logger.info(str(scheduler))
        return [optimizer], [scheduler]
    else:
        return optimizer
