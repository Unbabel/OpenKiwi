The $PYTHON environment variable should be set when running this test
from Python.

  $ [ -n "$PYTHON" ] || PYTHON="`which python`"
  $ [ -n "$PYTHONPATH" ] || PYTHONPATH="$TESTDIR/../" && export PYTHONPATH

  $ python $TESTDIR/../kiwi train --help
  usage: kiwi train [--config CONFIG] [--save-config SAVE_CONFIG] [-d] [-q]
                    [--log-interval LOG_INTERVAL]
                    [--mlflow-tracking-uri MLFLOW_TRACKING_URI]
                    [--experiment-name EXPERIMENT_NAME] [--run-uuid RUN_UUID]
                    [--output-dir OUTPUT_DIR]
                    [--mlflow-always-log-artifacts [MLFLOW_ALWAYS_LOG_ARTIFACTS]]
                    [--seed SEED] [--gpu-id GPU_ID] [--load-model LOAD_MODEL]
                    [--save-data SAVE_DATA] [--load-data LOAD_DATA]
                    [--load-vocab LOAD_VOCAB] [--epochs EPOCHS]
                    [--train-batch-size TRAIN_BATCH_SIZE]
                    [--valid-batch-size VALID_BATCH_SIZE]
                    [--optimizer {sgd,adagrad,adadelta,adam,sparseadam}]
                    [--learning-rate LEARNING_RATE]
                    [--learning-rate-decay LEARNING_RATE_DECAY]
                    [--learning-rate-decay-start LEARNING_RATE_DECAY_START]
                    [--checkpoint-validation-steps CHECKPOINT_VALIDATION_STEPS]
                    [--checkpoint-save [CHECKPOINT_SAVE]]
                    [--checkpoint-keep-only-best CHECKPOINT_KEEP_ONLY_BEST]
                    [--checkpoint-early-stop-patience CHECKPOINT_EARLY_STOP_PATIENCE]
                    [--resume [RESUME]] --model
                    {nuqe,estimator,predictor,quetch,linear}
  
  Args that start with '--' (eg. --save-config) can also be set in a config file
  (specified via --config). The config file uses YAML syntax and must represent
  a YAML 'mapping' (for details, see http://learn.getgrav.org/advanced/yaml). If
  an arg is specified in more than one place, then commandline values override
  config file values which override defaults.
  
  optional arguments:
    --config CONFIG       Load config file from path (default: None)
  
  I/O and logging:
    --save-config SAVE_CONFIG
                          Save parsed configuration and arguments to file
                          (default: None)
    -d, --debug           Output additional messages. (default: False)
    -q, --quiet           Only output warning and error messages. (default:
                          False)
    --log-interval LOG_INTERVAL
                          Log every k batches. (default: 100)
    --mlflow-tracking-uri MLFLOW_TRACKING_URI
                          Log model parameters, training metrics, and artifacts
                          (files) to this MLflow server. Uses the localhost by
                          default. (default: mlruns/)
    --experiment-name EXPERIMENT_NAME
                          MLflow will log this run under this experiment name,
                          which appears as a separate section in the UI. It will
                          also be used in some messages and files. (default:
                          None)
    --run-uuid RUN_UUID   If specified, MLflow will log metrics and params under
                          this ID. If it exists, the run status will change to
                          running. This ID is also used for creating this run's
                          output directory. (Run ID must be a 32-character hex
                          string) (default: None)
    --output-dir OUTPUT_DIR
                          Output several files for this run under this
                          directory. If not specified, a directory under "runs"
                          is created or reused based on the Run UUID. Files
                          might also be sent to MLflow depending on the
                          --mlflow-always-log-artifacts option. (default: None)
    --mlflow-always-log-artifacts [MLFLOW_ALWAYS_LOG_ARTIFACTS]
                          Always log (send) artifacts (files) to MLflow
                          artifacts URI. By default (false), artifacts are only
                          logged if MLflow is a remote server (as specified by
                          --mlflow-tracking-uri option). All generated files are
                          always saved in --output-dir, so it might be
                          considered redundant to copy them to a local MLflow
                          server. If this is not the case, set this option to
                          true. (default: False)
  
  random:
    --seed SEED           Random seed (default: 42)
  
  gpu:
    --gpu-id GPU_ID       Use CUDA on the listed devices (default: None)
  
  save-load:
    --load-model LOAD_MODEL
                          Directory containing a model.torch file to be loaded
                          (default: None)
    --save-data SAVE_DATA
                          Output dir for saving the preprocessed data files.
                          (default: None)
    --load-data LOAD_DATA
                          Input dir for loading the preprocessed data files.
                          (default: None)
    --load-vocab LOAD_VOCAB
                          Directory containing a vocab.torch file to be loaded
                          (default: None)
  
  training:
    --epochs EPOCHS       Number of epochs for training. (default: 50)
    --train-batch-size TRAIN_BATCH_SIZE
                          Maximum batch size for training. (default: 64)
    --valid-batch-size VALID_BATCH_SIZE
                          Maximum batch size for evaluating. (default: 64)
  
  training-optimization:
    --optimizer {sgd,adagrad,adadelta,adam,sparseadam}
                          Optimization method. (default: adam)
    --learning-rate LEARNING_RATE
                          Starting learning rate. Recommended settings: sgd = 1,
                          adagrad = 0.1, adadelta = 1, adam = 0.001 (default:
                          1.0)
    --learning-rate-decay LEARNING_RATE_DECAY
                          Decay learning rate by this factor. (default: 1.0)
    --learning-rate-decay-start LEARNING_RATE_DECAY_START
                          Start decay after this epoch. (default: 0)
  
  training-save-load:
    --checkpoint-validation-steps CHECKPOINT_VALIDATION_STEPS
                          Perform validation every X training batches. (default:
                          0)
    --checkpoint-save [CHECKPOINT_SAVE]
                          Save a training snapshot when validation is run.
                          (default: False)
    --checkpoint-keep-only-best CHECKPOINT_KEEP_ONLY_BEST
                          Keep only this number of saved snapshots; 0 will keep
                          all. (default: 0)
    --checkpoint-early-stop-patience CHECKPOINT_EARLY_STOP_PATIENCE
                          Stop training if evaluation metrics do not improve
                          after X validations; 0 disables this. (default: 0)
    --resume [RESUME]     Resume training a previous run. The --run-uuid (and
                          possibly --experiment-name) option must be specified.
                          Files are then searched under the "runs" directory. If
                          not found, they are downloaded from the MLflow server
                          (check the --mlflow-tracking-uri option). (default:
                          False)
  
  models:
    --model {nuqe,estimator,predictor,quetch,linear}
                          Use 'kiwi train --model <model> --help' for specific
                          model options. (default: None)
