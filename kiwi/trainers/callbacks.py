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

import heapq
import logging
import shutil
import threading
from pathlib import Path

import torch

from kiwi import constants as const
from kiwi.data.utils import save_predicted_probabilities

logger = logging.getLogger(__name__)


class EarlyStopException(StopIteration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Checkpoint:
    """Class for determining whether to evaluate / save the model.
    """

    def __init__(
        self,
        output_dir,
        checkpoint_save=False,
        checkpoint_keep_only_best=0,
        checkpoint_early_stop_patience=0,
        checkpoint_validation_steps=0,
    ):
        """

        Args:
            output_dir (Path): Required if checkpoint_save == True.
            checkpoint_save (bool):
                Save a training snapshot when validation is run.
            checkpoint_keep_only_best:
                Keep only this number of saved snapshots; 0 will keep all.
            checkpoint_early_stop_patience:
                Stop training if evaluation metrics do not improve after /X/
                validations; 0 disables this.
            checkpoint_validation_steps:
                Perform validation every /X/ training batches.
        """
        self.output_directory = Path(output_dir)
        self.validation_steps = checkpoint_validation_steps

        self.early_stop_patience = checkpoint_early_stop_patience

        self.save = checkpoint_save
        self.keep_only_best = checkpoint_keep_only_best
        if self.keep_only_best <= 0:
            self.keep_only_best = float('inf')

        # if self.save and not self.output_directory:
        #     logger.warning('Asked to save training snapshots, '
        #                    'but no output directory was specified.')
        #     self.save = False

        self.main_metric = None

        # This should be kept as a heap
        self.best_stats_summary = []
        self.stats_summary_history = []
        self._last_saved = 0
        self._validation_epoch = 0
        self._best_checkpoint_path = None

    def must_eval(self, epoch=None, step=None):
        if epoch is not None:
            return True
        if step is not None:
            return self.validation_steps and step % self.validation_steps == 0
        return False

    def must_save_best(self, stats):
        if self._validation_epoch <= self.keep_only_best:
            return True
        elif stats > self.worst_stats():
            return True

    def early_stopping(self):
        no_improvement = self._validation_epoch - self._last_saved
        return 0 < self.early_stop_patience <= no_improvement

    def __call__(self, trainer, valid_iterator, epoch=None, step=None):
        if self.must_eval(epoch=epoch, step=step):
            eval_stats_summary = trainer.eval_epoch(valid_iterator)
            eval_stats_summary.log()
            if trainer.scheduler:
                trainer.scheduler.step(eval_stats_summary.main_metric_value())

            saved_path = self.check_in(
                trainer, eval_stats_summary, epoch=epoch, step=step
            )
            if saved_path:
                predictions = trainer.predict(valid_iterator)
                if predictions is not None:
                    save_predicted_probabilities(saved_path, predictions)
            elif self.early_stopping():
                raise EarlyStopException(
                    'Early stopping training after {} validations '
                    'without improvements on the validation set'.format(
                        self.early_stop_patience
                    )
                )

    def check_in(self, trainer, stats, epoch=None, step=None):
        """Saves stat summary and handles checkpoint saving."""
        self._validation_epoch += 1
        self.stats_summary_history.append(stats)
        if self.save:
            if self.must_save_best(stats):
                self._last_saved = self._validation_epoch
                output_path = self.make_output_path(epoch=epoch, step=step)
                path_to_remove = self.push_to_heap(stats, output_path)
                event = trainer.save(output_path)
                self._best_checkpoint_path = output_path
                if path_to_remove:
                    self.remove_snapshot(path_to_remove, event)

                self.save_latest(trainer, saved_best=True)
                return output_path

            output_path = self.save_latest(trainer)
            return output_path
        return None

    def make_output_path(self, epoch=None, step=None):
        if epoch is not None:
            sub_dir = 'epoch_{}'.format(epoch)
        elif step is not None:
            sub_dir = 'step_{}'.format(step)
        else:
            sub_dir = 'epoch_unknown'
        return self.output_directory / sub_dir

    def push_to_heap(self, stats, output_path):
        """Push stats and output path to the heap."""

        path_to_remove = None
        # The second element (`-self._validation_epoch`) serves as a timestamp
        # to ensure that in case of a tie, the earliest model is saved.
        heap_element = (stats, -self._validation_epoch, output_path)
        if self._validation_epoch <= self.keep_only_best:
            heapq.heappush(self.best_stats_summary, heap_element)
        else:
            worst_stat = heapq.heapreplace(
                self.best_stats_summary, heap_element
            )
            path_to_remove = str(worst_stat[2])  # Worst output path
        return path_to_remove

    def remove_snapshot(self, path_to_remove, event=None):
        """Remove snapshot locally and in MLFlow."""

        def _remove_snapshot(path, event, message):
            if event:
                event.wait()
            logger.info(message)
            # ignores directory not empty error
            shutil.rmtree(str(path), ignore_errors=True)
            if event:
                event.clear()

        removal_message = 'Removing previous snapshot: {}'.format(
            path_to_remove
        )

        t = threading.Thread(
            target=_remove_snapshot,
            args=(path_to_remove, event, removal_message),
            daemon=True,
        )
        try:
            t.start()
            return t
        except FileNotFoundError as e:
            logger.exception(e)

    def best_stats_and_path(self):
        if self.best_stats_summary:
            stat, order, path = max(self.best_stats_summary)
            return stat, path
        return None, None

    def best_iteration_path(self):
        _, path = self.best_stats_and_path()
        return path

    def best_stats(self):
        stats, _ = self.best_stats_and_path()
        return stats

    def worst_stats(self):
        if self.best_stats_summary:
            return self.best_stats_summary[0][0]
        else:
            return None

    def best_model_path(self):
        path = self.output_directory / const.BEST_MODEL_FILE
        if path.exists():
            return path
        return self.best_iteration_path()

    def last_model_path(self):
        """Generates the path where the latest model should be saved.
        """
        path = self.output_directory / const.LAST_CHECKPOINT_FOLDER
        temp_path = self.output_directory / const.TEMP_LAST_CHECKPOINT_FOLDER
        return path, temp_path

    def save_latest(self, trainer, saved_best=False):
        """Saves latest checkpoint of the current model.
        In case a model was just saved due to being the best validation, saves a
        pointer instead of the full model. Returns path of saved model.
        """
        output_path, temp_path = self.last_model_path()
        # if best was saved, there's no need to save the entire model twice
        if saved_best:
            temp_path.mkdir(exist_ok=True)
            logging.info('Saving training state to {}'.format(temp_path))
            for file in Path(self._best_checkpoint_path).glob('*'):
                file_path = temp_path / file.name
                torch.save(file, str(file_path))
        else:
            trainer.save(temp_path)

        if output_path.exists():
            t = self.remove_snapshot(output_path)
            t.join()

        logger.info('Moving {} to {}'.format(temp_path, output_path))
        shutil.move(str(temp_path), str(output_path))

        return output_path

    def check_out(self):
        best_path = self.best_iteration_path()
        if best_path:
            self.copy_best_model(best_path, self.output_directory)

    @staticmethod
    def copy_best_model(model_dir, output_dir):
        model_path = model_dir / const.MODEL_FILE
        best_model_path = output_dir / const.BEST_MODEL_FILE
        logging.info('Copying best model to {}'.format(best_model_path))
        shutil.copy(str(model_path), str(best_model_path))
        return best_model_path
