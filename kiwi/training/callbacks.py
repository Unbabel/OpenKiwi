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
import textwrap

import numpy as np
from pytorch_lightning import Callback

logger = logging.getLogger(__name__)


class BestMetricsInfo(Callback):
    """Class for logging current training metrics along with the best so far."""

    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        verbose: bool = True,
        mode: str = 'auto',
    ):
        super().__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.verbose = verbose

        mode_dict = {
            'min': np.less,
            'max': np.greater,
            'auto': np.greater if 'acc' in self.monitor else np.less,
        }

        if mode not in mode_dict:
            logger.info(
                f'BestMetricsInfo mode {mode} is unknown, fallback to auto mode.'
            )
            mode = 'auto'

        self.monitor_op = mode_dict[mode]
        self.min_delta *= 1 if self.monitor_op == np.greater else -1

        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_epoch = -1
        self.best_metrics = {}

    def on_train_begin(self, trainer, pl_module):
        # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_epoch = -1
        self.best_metrics = {}

    def on_train_end(self, trainer, pl_module):
        if self.best_epoch > 0 and self.verbose > 0:
            metrics_message = textwrap.fill(
                ', '.join(
                    [
                        '{}: {:0.4f}'.format(k, v)
                        for k, v in self.best_metrics.items()
                        if k.startswith('val_')
                    ]
                ),
                width=80,
                initial_indent='\t',
                subsequent_indent='\t',
            )
            best_path = trainer.checkpoint_callback.best_model_path
            if not best_path:
                best_path = (
                    "model was not saved; check flags in Trainer if this is not "
                    "expected"
                )
            logger.info(
                f'Epoch {self.best_epoch} had the best validation metric:\n'
                f'{metrics_message} \n'
                f'\t({best_path})\n'
            )

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        current = metrics.get(self.monitor)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_epoch = trainer.current_epoch
            self.best_metrics = metrics.copy()  # Copy or it gets overwritten
            if self.verbose > 0:
                logger.info('Best validation so far.')
        else:
            metrics_message = textwrap.fill(
                ', '.join(
                    [
                        f'{k}: {v:0.4f}'
                        for k, v in self.best_metrics.items()
                        if k.startswith('val_')
                    ]
                ),
                width=80,
                initial_indent='\t',
                subsequent_indent='\t',
            )
            logger.info(
                f'Best validation so far was in epoch {self.best_epoch}:\n'
                f'{metrics_message} \n'
            )
