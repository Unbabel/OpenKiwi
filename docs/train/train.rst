Train Interface
===============
 API:
-------------

Run::

    import kiwi

    train_predest_config = 'experiments/examples/train_predest.yaml'
    kiwi.train(train_predest_config)

    train_nuqe_config = 'experiments/examples/train_nuqe.yaml'
    kiwi.train(train_nuqe_config)


CLI:
-----

Run::

    kiwi train --config {model_config_file} [OPTS]



**Note**: Args that start with '-\\-' (eg. -\\-save-config) can also be set in a config file
(specified via -\\-config). The config file uses YAML syntax and must represent
a YAML 'mapping' (for details, see http://learn.getgrav.org/advanced/yaml). If
an arg is specified in more than one place, then command line values override
config file values which override defaults.

For generic options see: :ref:`here <generic-flags>`

Model specific options can be seen here:

.. toctree::
   :maxdepth: 1

   train_nuqe
   train_predictor
   train_predictor_estimator
   train_quetch
   train_linear

.. _train-flags:

.. argparse::
   :module: kiwi.cli.pipelines.train
   :passparser:
   :func: train_opts
   :prog: kiwi train
