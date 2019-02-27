
## Quickstart

This document gives a quick overview of how to use OpenKiwi with the provided example configuration file.

To be able to run the examples, download the [WMT17 En-DE Word Level Quality Estimation data](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11372/LRT-1974/task2_en-de_training-dev.tar.gz) (you will need to provide your email address to receive a download link) and extract its contents into `data/WMT17`. To be able to pretrain the `predictor` model, additionally download the [English-German in-domain Corpus](https://www.quest.dcs.shef.ac.uk/wmt18_files_qe/corpus_en-de.tar.gz) provided by WMT and extract its contents into `data/WMT18`.


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
 - `linear` (http://www.aclweb.org/anthology/Q17-1015)

The Estimator model relies on pre-training of its component model `predictor`.

Options are handled via `yaml` configuration files.
On the command line, you can alternatively pass parameters as arguments,
or mix the two modes to override parameters in a config file.

### Train Pipeline

Before we continue into specific instructions for training models please take into account
that if you're working with data from WMT18 or later, you must make sure to pass the 
`--wmt18-format True` flag. Either through yaml, or the CLI. This happens because WMT18 
introduced `GAP` tags that classify the gaps between words.
Note that the example files provided use WMT17 data for the `predictor-estimator` and 
WMT18 data for `NuQE`.
You can check all the configuration options in :ref:`here <train-flags>`

#### Estimator Model

Example Configuration files with detailed hyperparameter documentation:
* [experiments/train_predictor.yaml](https://github.com/Unbabel/OpenKiwi/tree/master/experiments/train_predictor.yaml)
* [experiments/train_estimator.yaml](https://github.com/Unbabel/OpenKiwi/tree/master/experiments/train_estimator.yaml)

The Estimator Model performs best if initialized with a `predictor` model that is pretrained on parallel data.
The example configuration file requires ca. 24hrs to train 3 epochs on a GPU with 12GB RAM.
To train with less memory, change the `train-batch-size` and `valid-batch-size` parameters.
To train on CPU, comment out the line
```
gpu-id: 0
```

Train the `predictor` like so:
```python
import kiwi

predictor_config = experiments/train_predictor.yaml'
kiwi.train(predictor_config)
```
Or via CLI:
```bash
kiwi train --config experiments/train_predictor.yaml
```
After the run has finished, the best model is saved at (`--output-dir` option) `runs/predictor/best_model.torch`.
Now we can train the `estimator` model using the configuration `experiments/train_estimator.yaml`:
```python
import kiwi

estimator_config = 'experiments/train_estimator.yaml'
kiwi.train(estimator_config)
```
Or:
```bash
kiwi train --config experiments/train_estimator.yaml
```

When a pretrained `predictor` is provided in this way, the hyperparameters relating
to the `predictor` architecture as well as the vocabulary mapping are retrieved from the model.
Setting them to different values in the `estimator` config has no effect.


#### NuQE Model

Example Configuration file with detailed hyperparameter documentation:
* [experiments/train_nuqe.yaml](https://github.com/Unbabel/OpenKiwi/tree/master/experiments/train_nuqe.yaml)

Run via API:
```python
import kiwi

nuqe_config = 'experiments/train_nuqe.yaml'
kiwi.train(nuqe_config)
```
Or via CLI:
```bash
kiwi train --config experiments/train_nuqe.yaml
```


### Predict

The predict pipeline takes a trained  QE model as input and uses it to evaluate the quality of machine translations.
The API provides the function [load_model](source/kiwi.lib.html#kiwi.lib.predict.load_model) which returns a
[Predictor](source/kiwi.predictors.html#module-kiwi.predictors.predictor)  object that can be used to generate predictions.

To predict the quality of a set of machine translation dataset via the CLI,  use a `yaml` file specifying model path,
output directory, source and target data, and run:

```bash
kiwi predict --config experiments/predict_predest.yaml

kiwi predict --config experiments/predict_nuqe.yaml
```

which will generate predictions for the word and sentence level quality of the WMT18 en-de dev set in the folder `predictions/predest`.


### Evaluate

The evaluate pipeline takes predictions of a trained model and a reference (gold) file and evaluates the performance of the model based on the comparison between the predictions and the reference.

To evaluate one of your models via the CLI, create a `yaml` file specifying the format of predictions, format of reference and the location of these files, and run:

```bash
kiwi evaluate --config experiments/estimator_evaluate.yaml

kiwi evaluate --config experiments/nuqe_evaluate.yaml
```

You can check all the configuration options in :ref:`here <evaluate-flags>`

[//]: # (NOTE: please change docs/cli.rst after fixing Search)
