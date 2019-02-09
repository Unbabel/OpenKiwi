import logging
import threading

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def retrieve_mlflow_experiment_id(name, create=False):
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


class MLflowLogger:
    def __init__(self):
        self.always_log_artifacts = False
        self._experiment_name = None

    def is_remote(self):
        return not mlflow.tracking.utils._is_local_uri(
            mlflow.get_tracking_uri()
        )

    def configure(
        self,
        run_uuid,
        experiment_name,
        tracking_uri,
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

        experiment_id = retrieve_mlflow_experiment_id(
            experiment_name, create=create_experiment
        )

        return mlflow.start_run(
            run_uuid=run_uuid, experiment_id=experiment_id, nested=nest_run
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
        return self.always_log_artifacts or self.is_remote()

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
    def set_tracking_uri(uri):
        mlflow.set_tracking_uri(uri)

    @staticmethod
    def end_run():
        mlflow.end_run()


mlflow_logger = MLflowLogger()
