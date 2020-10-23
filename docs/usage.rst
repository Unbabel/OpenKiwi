Usage
=====

Kiwi's functionality is split in ``pipelines``::

   train, pretrain, predict, evaluate, search


Currently supported models are:

* NuQE (http://www.aclweb.org/anthology/Q17-1015)
* PredictorEstimator (http://www.aclweb.org/anthology/W17-4763)
* BERT
* XLM
* XLM-Roberta

All pipelines and models are configured via:

* YAML files, for CLI or API usage
* a dictionary, for API usage

The predicting pipeline additionally provides a simplified interface with explicit arguments.

For CLI usage, the general command is::

    kiwi (train|pretrain|predict|evaluate|search) CONFIG_FILE

Example configuration files can be found in ``config/``. Details are covered in
:ref:`configuration`, including how to override options in the CLI.


Training and pretraining
------------------------

The ``PredictorEstimator`` requires pretraining the Predictor model with parallel data.
This type of model is called a `TLM model`, for "Translation Language Model".

The BERT-based models are already pretrained. However, they can be fine-tuned before being
used for QE. To do so, it is currently necessary to use the original scripts, which can be
found in ``scripts/pre_finetuning_transformers``.

Examples of how to call Kiwi to train BERT for QE:

.. code-block:: bash

   kiwi train config/bert.yaml

Or:

.. code-block:: python

   from kiwi.lib.train import train_from_file

   run_info = train_from_file('config/bert.yaml')

Or:

.. code-block:: python

   from kiwi.lib.train import train_from_configuration
   from kiwi.lib.utils import file_to_configuration

   configuration_dict = file_to_configuration('config/bert.yaml')
   run_info = train_from_configuration(configuration_dict)


The ``configuration_dict`` is only validated inside ``train_from_configuration``, which
means other file formats can be used. In fact, ``file_to_configuration`` also supports JSON files
(but that is a not well known fact, as YAML is preferred).

Pretraining the ``Predictor`` can be down by calling:

.. code-block:: bash

   kiwi pretrain config/predictor.yaml


Predicting
----------

The predict pipeline takes a trained  QE model as input and uses it to evaluate the quality of machine translations.

The API provides  which returns a
`Predictor <source/kiwi.predictors.html#module-kiwi.predictors.predictor>`_  object that can be used to generate predictions.

To predict the quality of a set of machine translation dataset via the CLI, use a
``yaml`` file specifying model path, output directory, source and target data, etc.
(details are explained in :ref:`configuration`), and run::

   kiwi predict config/predict.yaml


As a package, there a few alternatives, depending on the use.

To load a trained model and produce predictions on a full dataset, use:

.. code-block:: python

   from kiwi.lib.predict import predict_from_configuration
   from kiwi.lib.utils import file_to_configuration

   configuration_dict = file_to_configuration('config/predict.yaml')
   predictions, metrics = predict_from_configuration(configuration_dict)


To load a trained model and keep it in memory for predicting on-demand, use:

.. code-block:: python

   from kiwi.lib.predict import load_system

   runner = load_system('trained_models/model.ckpt')
   predictions = runner.predict(
       source=['Aqui vai um exemplo de texto'],
       target=['Here is an example text'],
   )


The ``predictions`` object will contain one or more of the following attributes::

    sentences_hter
    target_tags_BAD_probabilities
    target_tags_labels
    source_tags_BAD_probabilities
    source_tags_labels
    gap_tags_BAD_probabilities
    gap_tags_labels


More details can be found in the code :ref:`reference`.


Evaluating
----------

The evaluate pipeline takes predictions of a trained model and a reference (gold) file
and evaluates the performance based on several metrics.

To evaluate one of your models via the CLI, create a ``yaml`` file specifying the format
of predictions, format of reference and the location of these files, and run:

.. code-block:: bash

   kiwi evaluate config/predict.yaml

Or alternatively:

.. code-block:: python

   from kiwi.lib.evaluate import evaluate_from_configuration
   from kiwi.lib.utils import file_to_configuration

   configuration_dict = file_to_configuration('config/evaluate.yaml')
   report = evaluate_from_configuration(configuration_dict)
   print(report)


You can check all the configuration options in :ref:`configuration`.


Searching
---------

The search pipeline enables hyperparameter search for the Kiwi models using the
`Optuna <https://github.com/optuna/optuna>`_ library.

Examples of how to call Kiwi to search hyperparameters for BERT for QE:

.. code-block:: bash

   kiwi search config/search.yaml

Or:

.. code-block:: python

   from kiwi.lib.search import search_from_file

   optuna_study = search_from_file('config/search.yaml')

Or:

.. code-block:: python

   from kiwi.lib.search import search_from_configuration
   from kiwi.lib.utils import file_to_configuration

   configuration_dict = file_to_configuration('config/search.yaml')
   optuna_study = search_from_configuration(configuration_dict)


The search configuration ``search.yaml`` points to the base training config
(``config/bert.yaml`` in the above BERT example) which defines the basic model,
and the rest of the options are dedicated to configuring the hyperparameters to search
and the ranges to search them in.
