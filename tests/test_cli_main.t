The $PYTHON environment variable should be set when running this test
from Python.

  $ [ -n "$PYTHON" ] || PYTHON="`which python`"
  $ [ -n "$PYTHONPATH" ] || PYTHONPATH="$TESTDIR/../" && export PYTHONPATH

  $ python $TESTDIR/../kiwi --help
  usage: kiwi [-h] [--version] {train,infer,search,jackknife,preprocess} ...
  
  Quality Estimation toolkit
  
  optional arguments:
    -h, --help            show this help message and exit
    --version             show program's version number and exit
  
  Pipelines:
    Use 'kiwi <pipeline> (-h | --help)' to check it out.
  
    {train,infer,search,jackknife,preprocess}
                          Available pipelines:
      train               Train a QE model
      infer             Use a pre-trained model for prediction
      search              Search training hyperparameters for a QE model
      jackknife           Jackknife training data with model
      preprocess          Preprocess and save a dataset
