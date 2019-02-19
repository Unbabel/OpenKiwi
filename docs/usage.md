
## Usage

Here we describe the basic usage steps.
Please refer to :ref:`configs` for a full run down of the options provided.

OpenKiwi can be used as a package from within Python or from the command line.

Its functionality is split in `pipelines`.
Currently supported are:
```
train, predict, jackknife, evaluate
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

[Example configuration files TODO INSERT LINK]() are provided and will be referred to throughout this introduction.
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
kiwi train --config {model_config_file} [OPTS]
```

You can check all the configuration options in :ref:`here <train-flags>`

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
kiwi train --config experiments/examples/train_predictor.yaml
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
kiwi train --config experiments/examples/train_estimator.yaml
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
kiwi train --config experiments/examples/train_nuqe.yaml
```


### Predict

The predict pipeline takes a trained model as input and uses it to generate predictions on new data.

The API provides the function :py:func:kiwi.load_model which returns a
:py:class:kiwi.predictors.predictor.Predictor object that can be used to generate predictions.

To generate predictions for a dataset via the CLI, create a `YAML` file specifying model path,
output directory, source and target data, and run:
```bash
kiwi predict --config experiments/examples/predict_predest.yaml
```

You can check all the configuration options in :ref:`here <predict-flags>`

### Evaluate

The evaluate pipeline takes predictions of a trained model and a reference (gold) file and evaluates the performance of the model based on the comparison between the predictions and the reference.

To generate evaluate one of your models via the CLI, create a `YAML` file specifying the format of predictions, format of reference and the location of these files, and run:

```bash
kiwi evaluate --config experiments/examples/evaluate_predest.yaml
```

You can check all the configuration options in :ref:`here <evaluate-flags>`

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
kiwi search --config experiment/examples/search_predest.yaml
```
Or using the API:
```python
import kiwi

search_config = 'experiments/example/search_predest.yaml'
kiwi.search(search_config)
```

This will do a full grid search on all parameters specified in `search_predest.yaml`,
using the values in `train_predest.yaml` as defaults.
