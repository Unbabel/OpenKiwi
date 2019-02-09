OpenKiwi Configuration
======================

OpenKiwi can be configured in a plethora of different ways. In order to allow flexibility in integration into different pipeline OpenKiwi can be used through a CLI (command line interface) or as a python module.

In both cases however, the configuration options can be passed into OpenKiwi in the form of a YAML config file.
Either by calling ```kiwi.train(config.yaml)``` or by passing the YAML on the CLI with ```--config```.
Below you can find both the instructions on how to interact with OpenKiwi through it's CLI and an extensive list of the different configuration options accepted by OpenKiwi.

.. toctree::
   :maxdepth: 3

   CLI
   flags
