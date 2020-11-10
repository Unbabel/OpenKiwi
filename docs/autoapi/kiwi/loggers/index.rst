:mod:`kiwi.loggers`
===================

.. py:module:: kiwi.loggers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   kiwi.loggers.MLFlowTrackingLogger



Functions
~~~~~~~~~

.. autoapisummary::

   kiwi.loggers.normalize_metric_key
   kiwi.loggers.validate_metric_value


.. data:: logger
   

   

.. data:: _INVALID_PARAM_AND_METRIC_CHARACTERS
   

   

.. function:: normalize_metric_key(key)

   Normalize key name for MLflow.

   mlflow.exceptions.MlflowException: Invalid metric name: 'WMT19_F1_MULT+PEARSON'.
   Names may only contain alphanumerics, underscores (_), dashes (-), periods (.),
   spaces ( ), and slashes (/).

   This is raised by matching against ``r"^[/\w.\- ]*$"``.


.. function:: validate_metric_value(value)


.. py:class:: MLFlowTrackingLogger(experiment_name: str = 'default', run_id: Optional[str] = None, tracking_uri: Optional[str] = None, tags: Optional[Dict[str, Any]] = None, save_dir: Optional[str] = None, always_log_artifacts: bool = False)

   Bases: :class:`pytorch_lightning.loggers.MLFlowLogger`

   Log using `MLflow <https://mlflow.org>`_. Install it with pip:

   .. code-block:: bash

       pip install mlflow

   .. rubric:: Example

   >>> from pytorch_lightning import Trainer
   >>> from pytorch_lightning.loggers import MLFlowLogger
   >>> mlf_logger = MLFlowLogger(
   ...     experiment_name="default",
   ...     tracking_uri="file:./ml-runs"
   ... )
   >>> trainer = Trainer(logger=mlf_logger)

   Use the logger anywhere in you :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

   >>> from pytorch_lightning import LightningModule
   >>> class LitModel(LightningModule):
   ...     def training_step(self, batch, batch_idx):
   ...         # example
   ...         self.logger.experiment.whatever_ml_flow_supports(...)
   ...
   ...     def any_lightning_module_function_or_hook(self):
   ...         self.logger.experiment.whatever_ml_flow_supports(...)

   :param experiment_name: The name of the experiment
   :param tracking_uri: Address of local or remote tracking server.
                        If not provided, defaults to `file:<save_dir>`.
   :param tags: A dictionary tags for the experiment.
   :param save_dir: A path to a local directory where the MLflow runs get saved.
                    Defaults to `./mlflow` if `tracking_uri` is not provided.
                    Has no effect if `tracking_uri` is provided.

   .. method:: tracking_uri(self)
      :property:


   .. method:: log_param(self, key, value)


   .. method:: log_hyperparams(self, params: Dict[str, Any]) -> None

      Record hyperparameters.

      :param params: :class:`~argparse.Namespace` containing the hyperparameters


   .. method:: log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix='') -> None

      Records metrics.
      This method logs metrics as as soon as it received them. If you want to aggregate
      metrics for one specific `step`, use the
      :meth:`~pytorch_lightning.loggers.base.LightningLoggerBase.agg_and_log_metrics` method.

      :param metrics: Dictionary with metric names as keys and measured quantities as values
      :param step: Step number at which the metrics should be recorded


   .. method:: log_artifact(self, local_path, artifact_path=None)


   .. method:: log_artifacts(self, local_dir, artifact_path=None)


   .. method:: get_artifact_uri(self)


   .. method:: log_model(self, local_file, name=None)


   .. method:: log_tag(self, name: str, value: str)


   .. method:: log_tags(self, tags: Dict[str, str])



