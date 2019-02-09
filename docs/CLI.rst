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
         Currently not supported.

   predict
         For prediction configuration options see :doc:`cli/predict`.

   jackknife
         For jackknife configuration options see :doc:`cli/jackknife`.

   train
         For train configuration options see :doc:`cli/train`.

   evaluate
         For evaluation configuration options see :doc:`cli/evaluate`.
