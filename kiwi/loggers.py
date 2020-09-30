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
import re
import threading
from typing import Any, Dict, Optional

import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only

from kiwi.utils.io import generate_slug

logger = logging.getLogger(__name__)

_INVALID_PARAM_AND_METRIC_CHARACTERS = re.compile(r'[^/\w.\- ]')


def normalize_metric_key(key):
    """Normalize key name for MLflow.

    mlflow.exceptions.MlflowException: Invalid metric name: 'WMT19_F1_MULT+PEARSON'.
    Names may only contain alphanumerics, underscores (_), dashes (-), periods (.),
    spaces ( ), and slashes (/).

    This is raised by matching against ``r"^[/\\w.\\- ]*$"``.
    """
    return _INVALID_PARAM_AND_METRIC_CHARACTERS.sub('-', key)


def validate_metric_value(value):
    if torch.is_tensor(value):
        return value.mean().item()
    else:
        try:
            return float(value)
        except TypeError:
            return None


class MLFlowTrackingLogger(MLFlowLogger):
    def __init__(
        self,
        experiment_name: str = 'default',
        run_id: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        always_log_artifacts: bool = False,
    ):
        super().__init__(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
        )
        self.always_log_artifacts = always_log_artifacts

        if run_id:
            run = self._mlflow_client.get_run(run_id)
            self._run_id = run.info.run_id
            self._experiment_id = run.info.experiment_id
        else:
            # Force creation of experiment and run
            _ = self.run_id

    @property
    def tracking_uri(self):
        from mlflow import get_tracking_uri

        return get_tracking_uri()
        # return self._mlflow_client._tracking_client.tracking_uri

    @rank_zero_only
    def log_param(self, key, value):
        self.experiment.log_param(self.run_id, key, value)

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params, delimiter='.')
        for k, v in params.items():
            self.experiment.log_param(self.run_id, k, v)

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None, prefix=''
    ) -> None:
        # FIXME: do this in the background (non-blocking way)
        normalized_metrics = {}
        for key, value in metrics.items():
            if prefix:
                key = '{}_{}'.format(prefix, key)
            key = normalize_metric_key(key)
            value = validate_metric_value(value)
            normalized_metrics[key] = value
        super().log_metrics(metrics=normalized_metrics, step=step)

    @rank_zero_only
    def log_artifact(self, local_path, artifact_path=None):
        t = threading.Thread(
            target=self.experiment.log_artifact,
            args=(self.run_id, local_path),
            kwargs={'artifact_path': artifact_path},
            daemon=True,
        )
        t.start()
        return t

    @rank_zero_only
    def log_artifacts(self, local_dir, artifact_path=None):
        def send(e, run_id, dpath, path):
            self.experiment.log_artifacts(run_id, dpath, artifact_path=path)
            e.set()

        event = threading.Event()
        t = threading.Thread(
            target=send,
            args=(event, self.run_id, local_dir, artifact_path),
            daemon=True,
        )
        t.start()
        return event

    def get_artifact_uri(self):
        run = self.experiment.get_run(self.run_id)
        return run.info.artifact_uri

    @rank_zero_only
    def log_model(self, local_file, name=None):
        from mlflow.exceptions import MlflowException

        if not name:
            name = generate_slug(self._experiment_name)

        try:
            self._mlflow_client.create_model_version(
                name=name, source=local_file, run_id=self.run_id
            )
        except MlflowException as e:
            logger.warning(e.message)
            logger.info('Logging model as artifact instead')
            thread = self.log_artifact(local_file, artifact_path='model')
            thread.join()

    @rank_zero_only
    def log_tag(self, name: str, value: str):
        self._mlflow_client.set_tag(self.run_id, name, value)

    @rank_zero_only
    def log_tags(self, tags: Dict[str, str]):
        for name, value in tags.items():
            self.log_tag(name, value)
