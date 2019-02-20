jackknife Interface
===================

Jackknifing is a procedure used to create "un-biased" predictions from a training set. It employs a system similar to K-fold Cross Validation where the dataset is divided into  K different slices. Then, a model is trained for each combination of K-1 slices and creates predictions for the remaining slice. This is a lengthy process as it requires K training runs of the model.

**Note**: Args that start with '-\\-' (eg. -\\-save-config) can also be set in a config file
(specified via -\\-config). The config file uses YAML syntax and must represent
a YAML 'mapping' (for details, see http://learn.getgrav.org/advanced/yaml). If
an arg is specified in more than one place, then command line values override
config file values which override defaults.

For generic options see: :ref:`here <generic-flags>`


.. _jackknife-flags:

.. argparse::
   :module: kiwi.cli.pipelines.jackknife
   :passparser:
   :func: jackknife_opts
   :prog: kiwi jackknife

