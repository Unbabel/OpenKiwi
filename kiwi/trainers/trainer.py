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
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

import kiwi
from kiwi import constants as const
from kiwi.loggers import tracking_logger
from kiwi.metrics.stats import Stats
from kiwi.models.model import Model
from kiwi.models.utils import load_torch_file
from kiwi.trainers.callbacks import EarlyStopException
from kiwi.trainers.utils import optimizer_class

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self, model, optimizer, checkpointer, log_interval=100, scheduler=None
    ):
        """
        Args:
          model: A kiwi.Model to train
          optimizer: An optimizer
          checkpointer: A Checkpointer object
          log_interval: Log train stats every /n/ batches. Default 100
          scheduler: A learning rate scheduler
        """
        self.model = model
        self.stats = Stats(
            metrics=model.metrics(),
            main_metric_ordering=model.metrics_ordering(),
            log_interval=log_interval,
        )

        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.scheduler = scheduler
        self._step = 0
        self._epoch = 0

    @property
    def stats_summary_history(self):
        return self.checkpointer.stats_summary_history

    def run(self, train_iterator, valid_iterator, epochs=50):
        """

        Args:
            train_iterator:
            epochs: Number of epochs for training.
        """
        # log(self.eval_epoch(valid_dataset))
        for epoch in range(self._epoch + 1, epochs + 1):
            logger.info('Epoch {} of {}'.format(epoch, epochs))
            self.train_epoch(train_iterator, valid_iterator)
            self.stats.log()
            try:
                self.checkpointer(self, valid_iterator, epoch=epoch)
            except EarlyStopException as e:
                logger.info(e)
                break

        self.checkpointer.check_out()

    def train_epoch(self, train_iterator, valid_iterator):
        self.model.train()
        for batch in tqdm(
            train_iterator,
            total=len(train_iterator),
            desc='Batches',
            unit=' batches',
            ncols=80,
        ):
            self._step += 1
            outputs = self.train_step(batch)
            self.stats.update(batch=batch, **outputs)
            self.stats.log(step=self._step)
            try:
                self.checkpointer(self, valid_iterator, step=self._step)
            except EarlyStopException as e:
                logger.info(e)
                break
        self._epoch += 1

    def train_steps(self, train_iterator, valid_iterator, max_steps):
        train_iterator.repeat = True
        self.model.train()
        step = 0
        for step, batch in tqdm(
            enumerate(train_iterator, 1),
            total=max_steps,
            desc='Steps',
            unit=' batches',
            ncols=80,
        ):
            self._step += 1
            outputs = self.train_step(batch)
            self.stats.update(batch=batch, **outputs)
            self.stats.log(step=self._step)
            try:
                self.checkpointer(self, valid_iterator, step=self._step)
            except EarlyStopException as e:
                logger.info(e)
                break

            if step > max_steps:
                break

        eval_stats_summary = self.eval_epoch(valid_iterator)
        eval_stats_summary.log()

        sub_path = Path('step_{}'.format(self._step))
        self.save(self.checkpointer.output_directory / sub_path)

        train_iterator.repeat = False

    def train_step(self, batch):
        self.model.zero_grad()
        model_out = self.model(batch)
        loss_dict = self.model.loss(model_out, batch)
        loss_dict[const.LOSS].backward()
        self.optimizer.step()
        return dict(loss=loss_dict, model_out=model_out)

    def eval_epoch(self, valid_iterator, prefix='EVAL'):
        self.model.eval()
        self.stats.reset()
        with torch.no_grad():
            for batch in valid_iterator:
                outputs = self.eval_step(batch)
                self.stats.update(batch=batch, **outputs)
        stats_summary = self.stats.wrap_up(prefix=prefix)
        self.model.train()
        return stats_summary

    def eval_step(self, batch):
        model_out = self.model(batch)
        loss_dict = self.model.loss(model_out, batch)
        return dict(loss=loss_dict, model_out=model_out)

    def predict(self, valid_iterator):
        self.model.eval()
        with torch.no_grad():
            predictions = defaultdict(list)
            for batch in valid_iterator:
                model_pred = self.model.predict(batch)
                for key, values in model_pred.items():
                    predictions[key] += values
        self.model.train()
        return predictions

    def make_sub_directory(self, root_directory, current_epoch, prefix='epoch'):
        root_path = Path(root_directory)
        epoch_path = Path('{}_{}'.format(prefix, current_epoch))
        output_path = root_path / epoch_path
        output_path.mkdir(exist_ok=True)
        return output_path

    def save(self, output_directory):
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True)
        logging.info('Saving training state to {}'.format(output_directory))

        model_path = output_directory / const.MODEL_FILE
        self.model.save(str(model_path))

        optimizer_path = output_directory / const.OPTIMIZER
        scheduler_dict = None
        if self.scheduler:
            scheduler_dict = {
                'name': type(self.scheduler).__name__.lower(),
                'state_dict': self.scheduler.state_dict(),
            }
        optimizer_dict = {
            'name': type(self.optimizer).__name__.lower(),
            'state_dict': self.optimizer.state_dict(),
            'scheduler_dict': scheduler_dict,
        }
        torch.save(optimizer_dict, str(optimizer_path))

        state = {
            '__version__': kiwi.__version__,
            '_epoch': self._epoch,
            '_step': self._step,
            'checkpointer': self.checkpointer,
        }
        state_path = output_directory / const.TRAINER
        torch.save(state, str(state_path))

        # Send to MLflow
        event = None
        if tracking_logger.should_log_artifacts():
            logger.info('Logging artifacts to {}'.format(output_directory))
            event = tracking_logger.log_artifacts(
                str(output_directory), artifact_path=str(output_directory.name)
            )
        return event

    @classmethod
    def from_directory(cls, directory, device_id=None):
        logger.info('Loading training state from {}'.format(directory))
        root_path = Path(directory)

        model_path = root_path / const.MODEL_FILE
        model = Model.create_from_file(model_path)

        if device_id is not None:
            model.to(device_id)

        optimizer_path = root_path / const.OPTIMIZER
        optimizer_dict = load_torch_file(str(optimizer_path))

        optimizer = optimizer_class(optimizer_dict['name'])(
            model.parameters(), lr=0.0
        )
        optimizer.load_state_dict(optimizer_dict['state_dict'])

        trainer = cls(model, optimizer, checkpointer=None)
        trainer_path = root_path / const.TRAINER
        state = load_torch_file(str(trainer_path))
        trainer.__dict__.update(state)
        return trainer

    @classmethod
    def resume(cls, local_path=None, prefix='latest_', device_id=None):
        if local_path:
            artifacts_uri = Path(local_path)
        else:
            artifacts_uri = Path(tracking_logger.get_artifact_uri())

        if Path(local_path) / Path(prefix + 'epoch') in artifacts_uri.glob(
            '{}*'.format(prefix)
        ):
            last_save = 'epoch'

        else:
            logging.info(
                'Latest epoch not found. Looking for other checkpoints'
            )
            prefix = 'epoch_'
            saved_checkpoints = [
                int(str(path.name).replace(prefix, ''))
                for path in artifacts_uri.glob('{}*'.format(prefix))
                if path.is_dir()
            ]
            if not saved_checkpoints:
                raise FileNotFoundError(
                    "Couldn't load trainer from: {}".format(
                        artifacts_uri / (prefix + '*')
                    )
                )
            last_save = max(saved_checkpoints)

        snapshot_dir = artifacts_uri / '{}{}'.format(prefix, last_save)
        logger.info('Resuming training from: {}'.format(snapshot_dir))
        return cls.from_directory(snapshot_dir, device_id=device_id)
