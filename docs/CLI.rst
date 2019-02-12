.. _command-line:

Command Line Interface
======================

The CLI allows OpenKiwi to carry it's pipelines without having to import kiwi as a python module.
After calling one of these pipelines, the specific parameters can be configured by using the flags available in the :ref:`configuration options <configs>`

.. argparse::
   :module: kiwi.cli.main
   :func: build_parser
   :prog: kiwi


   preprocess
         Deprecated.


   search
         Deprecated.

   predict
         For prediction configuration options see :doc:`predict/predict`

   jackknife
         For jackknife configuration options see :doc:`jackknife/jackknife`

   train
         For train configuration options see :doc:`train/train`
