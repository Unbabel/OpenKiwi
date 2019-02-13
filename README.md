# OpenKiwi

[![pipeline status](https://gitlab.com/Unbabel/OpenKiwi/badges/master/pipeline.svg)](https://gitlab.com/Unbabel/OpenKiwi/commits/master)
[![coverage report](https://gitlab.com/Unbabel/OpenKiwi/badges/master/coverage.svg)](https://gitlab.com/Unbabel/OpenKiwi/commits/master)

Toolkit for Machine Translation Quality Estimation.

## Features

* Framework for training QE models and using pre-trained models for evaluating MT
* Easy to use API. Import it as a package in other projects or run from the command line
* Implementation of state-of-the-art QE systems from 2016, 2017, and matching performance with 2018
* Track your experiments with MLflow
* Great for sandboxing with new models


## Results

http://www.statmt.org/wmt18/quality-estimation-task.html#task2_results

English-Latvian (SMT), words in MT, test set:

| Model                   |   xF1   |
| ----------------------- | ------- |
| NuQE model 1            | 0.3746  |
| SHEF-PT (Best in WMT18) | 0.3608  |


English-Latvian (NMT), words in MT, test set:

| Model                  |   xF1   |
| ---------------------- | ------- |
| NuQE model 1           | 0.4377  |
| Conv64 (Best in WMT18) | 0.4293  |


## Installing

Please note that since Python>=3.5 is required, all the below commands, especially `pip`,
also have to be the Python 3 version. This might require that you run `pip3` instead.

This project uses a newer configuration format defined by PEP-518, namely, a `pyproject.toml` file.
In order to support it, we use [Poetry](https://github.com/sdispater/poetry) as the build system
and the dependency manager.

Since we want to allow OpenKiwi to be used both as an application and as a library,
this has the added benefit of allowing us to specify dependencies in a single location
and simplifying the packaging process. 
Consequently, you'll notice there's no `requirements.txt` and no `setup.py` files.
The alternative routes are explained below.


### For Local Development

Install Poetry via the recommended way:
```bash
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
```
It's also possible to use pip:
```bash
pip install poetry
```

The usual practice of using a virtual environment still applies, possibly also by installing
a local Python interpreter through [pyenv](https://github.com/pyenv/pyenv).

If you don't have Python 3.5, 3.6, or 3.7, or need help creating a virtualenv, check online guides
like [this](https://realpython.com/python-virtual-environments-a-primer/).

**Note**: There's currently an issue with poetry not detecting conda virtual environments. As such
we suggest using virtualenv.

After cloning this repository and creating and activating a virtualenv, instead of the traditional
`pip install -r requirements.txt`, just run
```bash
poetry install
```
to install all dependencies.

Then add the package directory to your `PYTHONPATH` to be able to run it
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Running
```bash
python kiwi
```
should now show you a help message.

Go on to [Usage](#usage) for further instructions.


### As a Package

To install OpenKiwi as a package, simply run
```bash
pip install openkiwi
```

You can now
```python
import kiwi
```
inside your project or run the command
```bash
kiwi
```

Go on to [Usage](#usage) for further instructions.


## Usage

Here we describe the basic usage steps.
Please refer to the documentation **(TODO place link here)** for a full run down of the options provided.

OpenKiwi can be used as a package from within Python or from the command line.

Its functionality is split in `pipelines`.
Currently supported are:
```
train, predict, jackknife
```

The models supported are:
 - `predictor` and `estimator` (http://www.aclweb.org/anthology/W17-4763)
 - `nuqe` (http://www.aclweb.org/anthology/Q17-1015)
 - `quetch` (http://aclweb.org/anthology/W15-3037)
 - `linear`

The Predictor-Estimator model relies on the pre-training of its component model `predictor`.

Options are handled via `YAML` configuration files.
On the command line, you can alternatively pass parameters as arguments,
or mix the two modes to override parameters in a config file.

[Example configuration files](experiments/examples) are provided and will be referred to throughout this introduction.
To be able to run the examples, download the [WMT17 En-DE Word Level Quality Estimation data](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11372/LRT-1974/task2_en-de_training-dev.tar.gz)
(you will need to provide your email address to receive a download link) and extract its contents into `testdata/WMT17`.

To be able to pretrain the `predictor` model, additionally download the [English-German in-domain Corpus](https://www.quest.dcs.shef.ac.uk/wmt18_files_qe/corpus_en-de.tar.gz)
provided by WMT and extract its contents into `testdata/parallel`.


### General Options

* `config`:  
The path to the configuration file. Within the `search` pipeline, path to base config specifying the parameters that stay fixed.

* `model`:  
The model name.

* `gpu-id`:  
GPU ID. Do not set or set to negative number for CPU training. Default: `None`

* `load-model`:  
Path to a trained model. Default: `None`


### Train

Unsurprisingly, it is used to `train` a model.

* API:

```python

import kiwi

train_predest_config = 'experiments/examples/train_predest.yaml'
kiwi.train(train_predest_config)

train_nuqe_config = 'experiments/examples/train_nuqe.yaml'
kiwi.train(train_nuqe_config)
```

* CLI:  

```bash
python kiwi train --config {model_config_file} [OPTS]
```

#### Available parameters

* `epochs`  
Number of epochs to train. Default: `50`

* `train-batch-size`  
Batch size on train. Default: `32`

* `valid-batch-size`  
Batch size on validation. Default: `32`

* `learning-rate`  
Default: `1.0`

* `checkpoint-save`  
Flag. If true, save model checkpoint if needed after each evaluation. If false, never save the model. Default: `False`


#### Predictor-Estimator Model

Example Configuration files with detailed hyperparameter documentation:
* [`experiments/examples/train_predictor.yaml`](experiments/examples/train_predictor.yaml)
* [`experiments/examples/train_estimator.yaml`](experiments/examples/train_estimator.yaml)

The Predictor Estimator Model performs best if initialized with a `predictor` model that is pretrained on parallel data.
The example configuration file requires ca. 24hrs to train 3 epochs on a GPU with 12GB RAM.
To train with less memory, change the `train-batch-size` and `valid-batch-size` parameters.
To train on CPU, comment out the line
```
gpu-id: 0
```

Train the `predictor` like so:
```python
import kiwi

predictor_config = 'experiments/examples/train_predictor.yaml'
kiwi.train(predictor_config)
```
Or:
```bash
python kiwi train --config experiments/examples/train_predictor.yaml
```
At the beginning of the run, the local path `runs/experiment_id/run_uuid` for logging will be printed.
The output model will be saved in
```bash
runs/experiment_id/run_uuid/best_model.torch
```
To load the pretrained `predictor` model when training the `estimator` model,
set the parameter `load-pred-target` in `experiments/examples/train_estimator.yaml`
to point to the correct location (we will add a programmatic way of doing this).

If a pretrained `predictor` is provided in this way, the hyperparameters relating
to the `predictor` architecture as well as the vocabulary mapping are retrieved from the model.
Setting them to different values in the `estimator` config has no effect.

Finally train the `estimator` model:
```python
import kiwi

estimator_config = 'experiments/examples/train_estimator.yaml'
kiwi.train(estimator_config)
```
Or:
```bash
python kiwi train --config experiments/examples/train_estimator.yaml
```


#### NuQE Model

Example Configuration file with detailed hyperparameter documentation:
* [`experiments/examples/train_nuqe.yaml`](experiments/examples/train_nuqe.yaml)

Run via API:
```python
import kiwi

nuqe_config = 'experiments/examples/train_nuqe.yaml'
kiwi.train(nuqe_config)
```
Or via CLI:
```bash
python kiwi train --config experiments/examples/train_nuqe.yaml
```


### Predict

The predict pipeline takes a trained model as input and uses it to generate predictions on new data.

The API provides the function :py:func:kiwi.load_model which returns a
:py:class:kiwi.predictors.predictor.Predictor object that can be used to generate predictions.

To generate predictions for a dataset via the CLI, create a `YAML` file specifying model path,
output directory, source and target data, and run:
```bash
python kiwi predict --config experiments/examples/predict_predest.yaml
```

[//]: # (NOTE: please change docs/cli.rst after fixing Search)

### Search - CURRENTLY NOT SUPPORTED

The search pipeline is an overlay on top of train that supports hyperparameter search.
To use it, create a `YAML` file specifying a list for each parameter that you want to search over.
Then, set the `config` parameter to point to a base config defining all fixed parameters
(if the base config specifies values for parameters redefined in the `search` config, they will be ignored).

Example (`experiments/example/search_predest.yaml`):
```yaml
config: experiments/examples/train_predest.yaml

bad-weight: [1.0, 2.0, 3.0]
hidden-size-est: [50, 100, 200]
experiment-name: Predictor Estimator Example Grid Search
```

Then run the pipeline via CLI:
```bash
python kiwi search --config experiment/examples/search_predest.yaml
```
Or using the API:
```python
import kiwi

search_config = 'experiments/example/search_predest.yaml'
kiwi.search(search_config)
```

This will do a full grid search on all parameters specified in `search_predest.yaml`,
using the values in `train_predest.yaml` as defaults.
