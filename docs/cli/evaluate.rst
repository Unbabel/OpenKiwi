Evaluate Interface
==================

**Note**:Args that start with '-\\-' (eg. -\\-save-config) can also be set in a config file
(specified via -\\-config). The config file uses YAML syntax and must represent
a YAML 'mapping' (for details, see http://learn.getgrav.org/advanced/yaml). If
an arg is specified in more than one place, then command line values override
config file values which override defaults.

.. _evaluate-flags:

.. argparse::
   :module: kiwi.cli.pipelines.evaluate
   :passparser:
   :func: evaluate_opts
   :prog: kiwi evaluate

