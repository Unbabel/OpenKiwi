The $PYTHON environment variable should be set when running this test
from Python.

  $ [ -n "$PYTHON" ] || PYTHON="`which python`"
  $ [ -n "$PYTHONPATH" ] || PYTHONPATH="$TESTDIR/../" && export PYTHONPATH

  $ python $TESTDIR/../kiwi jackknife --help
  usage: kiwi jackknife [--config CONFIG] [--splits SPLITS]
                        [--train-config TRAIN_CONFIG]
  
  Args that start with '--' (eg. --splits) can also be set in a config file
  (specified via --config). The config file uses YAML syntax and must represent
  a YAML 'mapping' (for details, see http://learn.getgrav.org/advanced/yaml). If
  an arg is specified in more than one place, then commandline values override
  config file values which override defaults.
  
  optional arguments:
    --config CONFIG       Load config file from path (default: None)
  
  jackknifing:
    --splits SPLITS       Jackknife with X folds. (default: 5)
    --train-config TRAIN_CONFIG
                          Path to config file with model parameters. (default:
                          None)
