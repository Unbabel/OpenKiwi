Predict Interface
=================

**Note**:Args that start with '-\\-' (eg. -\\-save-config) can also be set in a config file
(specified via -\\-config). The config file uses YAML syntax and must represent
a YAML 'mapping' (for details, see http://learn.getgrav.org/advanced/yaml). If
an arg is specified in more than one place, then command line values override
config file values which override defaults.

For generic options see: :ref:`here <generic-flags>`

Model specific options can be seen here:

.. toctree::
   :maxdepth: 1

   predict_nuqe
   predict_predictor
   predict_predictor_estimator
   predict_quetch
   predict_linear


.. argparse::
   :module: kiwi.cli.pipelines.predict
   :passparser:
   :func: predict_opts
   :prog: kiwi predict

