.. _configuration:

Configuration
=============

.. toctree::
   :maxdepth: 5
   :hidden:

   train
   predict
   evaluate
   search


Kiwi can be configured essentially by using dictionaries. To persist configuration
dictionaries, we recommend using YAML. You can find standard configuration files
in ``config/``.


CLI overrides
-------------

To run Kiwi with a configuration file but overriding a value, use the format
``key=value``, where nested keys can be encoded by using a dot notation.
For example::

    kiwi train config/bert.yaml trainer.gpus=0 system.batch_size=16


Configuration Composing
-----------------------

Kiwi uses Hydra/OmegaConf to compose configuration coming from different places.
This makes it possible to split configuration across multiple files.

In most files in ``config/``, like ``config/predict.yaml``, you'll notice this:

.. code-block:: yaml

   defaults:
    - data: wmt19.qe.en_de

This means the file ``config/data/wmt19.qe.en_de.yaml`` will be loaded into the
configuration found in ``config/predict.yaml``. **Notice** that ``wmt19.qe.en_de.yaml``
must use fully qualified keys levels, that is, the full nesting of keys.

The nice use case for this is allowing dynamically changing parts of the configuration.
For example, we can use::

    kiwi train config/bert.yaml data=unbabel.qe.en_pt

to use a different dataset (where ``config/data/unbabel.qe.en_pt.yaml`` contains the
configuration for the data files).
