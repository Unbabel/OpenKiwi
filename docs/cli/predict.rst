Predict Interface
=================

**Note**:Args that start with '-\\-' (eg. -\\-save-config) can also be set in a config file
(specified via -\\-config). The config file uses YAML syntax and must represent
a YAML 'mapping' (for details, see http://learn.getgrav.org/advanced/yaml). If
an arg is specified in more than one place, then command line values override
config file values which override defaults.

For generic options see: :ref:`here <generic-flags>`

When using the prediction interface you need to pass a pre-trained model using the `--load-model` flag described in :ref:`General options <generic-flags>`.
If you do not specify a pre-trained model to be loaded and used for predictions, OpenKiwi will not be able to predict anything.

Model specific options can be seen here:

.. toctree::
   :maxdepth: 1

   predict_nuqe
   predict_predictor_estimator
   predict_quetch
   predict_linear

.. _predict-flags:

.. argparse::
   :module: kiwi.cli.pipelines.predict
   :passparser:
   :func: predict_opts
   :prog: kiwi predict

