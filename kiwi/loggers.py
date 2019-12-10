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
import threading
import uuid

logger = logging.getLogger(__name__)


class TrackingLogger:
    class ActiveRun:
        def __init__(self, run_uuid, experiment_id):
            self.run_uuid = run_uuid
            self.experiment_name = experiment_id

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return exc_type is None

    def __init__(self):
        self._experiment_id = None
        self._experiment_name = None
        self._active_run_uuids = []

    def configure(
        self, run_uuid, experiment_name,
        run_name=None, nest_run=True, *args, **kwargs
    ):
        if len(self._active_run_uuids) > 0 and not nest_run:
            raise Exception(
                (
                    "A run is already active. To start a nested run, call "
                    "start_nested_run(), or configure() with nest_run=True"
                )
            )
        if not self._active_run_uuids:
            self._experiment_name = experiment_name
            self._experiment_id = 0
            self._run_name = run_name
        if run_uuid is None:
            self._active_run_uuids.append(uuid.uuid4().hex)
        else:
            self._active_run_uuids.append(run_uuid)

        return TrackingLogger.ActiveRun(
            run_uuid=self._active_run_uuids[-1],
            experiment_id=self._experiment_id,
        )

    def start_nested_run(self, run_name=None):
        return self.configure(
            run_uuid=run_name, experiment_name=None, nest_run=True
        )

    @property
    def run_uuid(self):
        return self._active_run_uuids[-1] if self._active_run_uuids else None

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def run_name(self):
        return self._run_name

    def should_log_artifacts(self):
        return False

    def get_tracking_uri(self):
        return None

    @staticmethod
    def log_metric(key, value):
        pass

    @staticmethod
    def log_param(key, value):
        pass

    @staticmethod
    def log_artifact(local_path, artifact_path=None):
        pass

    @staticmethod
    def log_artifacts(local_dir, artifact_path=None):
        return None

    @staticmethod
    def get_artifact_uri():
        return None

    @staticmethod
    def end_run():
        pass


class MLflowLogger:
    def __init__(self):
        self.always_log_artifacts = False
        self._experiment_name = None
        self._run_name = None

    def configure(
        self,
        run_uuid,
        experiment_name,
        tracking_uri,
        run_name=None,
        always_log_artifacts=False,
        create_run=True,
        create_experiment=True,
        nest_run=True,
    ):
        if mlflow.active_run() and not nest_run:
            logger.info('Ending previous MLFlow run: {}.'.format(self.run_uuid))
            mlflow.end_run()

        self.always_log_artifacts = always_log_artifacts
        self._experiment_name = experiment_name
        self._run_name = run_name

        # MLflow specific
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if run_uuid:
            existing_run = MlflowClient().get_run(run_uuid)
            if not existing_run and not create_run:
                raise FileNotFoundError(
                    'Run ID {} not found under {}'.format(
                        run_uuid, mlflow.get_tracking_uri()
                    )
                )

        experiment_id = self._retrieve_mlflow_experiment_id(
            experiment_name, create=create_experiment
        )
        return mlflow.start_run(
            run_uuid, experiment_id=experiment_id,
            run_name=run_name, nested=nest_run
        )

    def start_nested_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name, nested=True)

    @property
    def run_uuid(self):
        return mlflow.tracking.fluent.active_run().info.run_uuid

    @property
    def experiment_id(self):
        return mlflow.tracking.fluent.active_run().info.experiment_id

    @property
    def experiment_name(self):
        # return MlflowClient().get_experiment(self.experiment_id).name
        return self._experiment_name

    def should_log_artifacts(self):
        return self.always_log_artifacts or self._is_remote()

    @staticmethod
    def get_tracking_uri():
        return mlflow.get_tracking_uri()

    @staticmethod
    def log_metric(key, value):
        mlflow.log_metric(key, value)

    @staticmethod
    def log_param(key, value):
        mlflow.log_param(key, value)

    @staticmethod
    def log_artifact(local_path, artifact_path=None):
        t = threading.Thread(
            target=mlflow.log_artifact,
            args=(local_path,),
            kwargs={'artifact_path': artifact_path},
            daemon=True,
        )
        t.start()

    @staticmethod
    def log_artifacts(local_dir, artifact_path=None):
        def send(dpath, e, path):
            mlflow.log_artifacts(dpath, artifact_path=path)
            e.set()

        event = threading.Event()
        t = threading.Thread(
            target=send, args=(local_dir, event, artifact_path), daemon=True
        )
        t.start()
        return event

    @staticmethod
    def get_artifact_uri():
        return mlflow.get_artifact_uri()

    @staticmethod
    def end_run():
        mlflow.end_run()

    def _is_remote(self):
        return not mlflow.tracking.utils._is_local_uri(
            mlflow.get_tracking_uri()
        )

    @staticmethod
    def _retrieve_mlflow_experiment_id(name, create=False):
        experiment_id = None
        if name:
            existing_experiment = MlflowClient().get_experiment_by_name(name)
            if existing_experiment:
                experiment_id = existing_experiment.experiment_id
            else:
                if create:
                    experiment_id = mlflow.create_experiment(name)
                else:
                    raise Exception(
                        'Experiment "{}" not found in {}'.format(
                            name, mlflow.get_tracking_uri()
                        )
                    )
        return experiment_id


try:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_logger = MLflowLogger()
except ImportError:
    tracking_logger = TrackingLogger()
