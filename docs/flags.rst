.. _configs:

Configuration Options
=====================

OpenKiwi supports an extensive range of options through its command line interface. (or see :ref:`note <note>` below). All commands are prepended by a ```<pipeline>``` command. For the available pipelines see: :ref:`CLI <command-line>`


.. _note:

**Note**: Args that start with '-\\-' (eg. -\\-save-config) can also be set in a config file
(specified via -\\-config). The config file uses YAML syntax and must represent
a YAML 'mapping' (for details, see http://learn.getgrav.org/advanced/yaml). If
an arg is specified in more than one place, then command line values override
config file values which override defaults.

For pipeline specific options see:

.. toctree::
   :maxdepth: 1

   cli/train
   cli/predict
   cli/jackknife
   cli/evaluate


.. _generic-flags:

General Options
---------------

These options are pipeline independent and are available in all different pipelines.
They are divided into three different categories, general, IO and save/load.

.. argparse::
   :module: kiwi.cli.opts
   :passparser:
   :func: general_opts
   :prog: kiwi <pipeline>

.. argparse::
   :module: kiwi.cli.opts
   :passparser:
   :func: io_opts
   :prog: kiwi <pipeline>

.. argparse::
   :module: kiwi.cli.opts
   :passparser:
   :func: save_load_opts
   :prog: kiwi <pipeline>

.. argparse::
   :module: kiwi.cli.opts
   :passparser:
   :func: logging_opts
   :prog: kiwi <pipeline>
