Using pre-trained models
========================

Reproducing benchmark values
----------------------------

The values provided in the table presented in the README are the results of our models in the WMT18 test set. These values can only be obtained by submiting your model to the online leaderboard (`here <https://competitions.codalab.org/competitions/19306#results>`_). However, contrary to the test set, the training and dev sets for WMT18 are widely available and can be downloaded from their website.
As such, here we provide the results that these same models achieve in the dev set:



In order to reproduce the values provided in the table above you should follow the following steps:

#. :ref:`Download Data <download-wmt>`
#. :ref:`Download Models <download-model>`
#. :ref:`Setup Directory <setup-reproduce>`
#. :ref:`Get Numbers <scripts-reproduce>`

.. _download-wmt:

Downloading Data
----------------

In order to download the data necessary for running these pre-trained models please refer to `WMT18 Download <https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-2619>`_. Here you should be able to download both the training and dev sets for all relevant language pairs. 

.. _download-model:

Downloading Models
------------------

To download models, please refer to `Github Releases <https://github.com/Unbabel/OpenKiwi/releases>`_. From there, pick the models whose numbers you'd like to reproduce and download the appropriate zip archives.

.. _setup-reproduce:

Setup Directories
-----------------

In order for our script to know where to find the downloaded models and data we ask you to reproduce the following directory structure::

   example_folder
   |-- data
   |   `-- word-level
   |       |-- en_de.nmt
   |       |-- en_de.smt
   `-- models
       |-- reproduce_numbers.sh
       |-- model1
       |-- model2
       `-- modeln

Create a folder such as `example_folder` and move both the data and models folders inside that folder.

.. _scripts-reproduce:

The numbers
-----------

Finally, we prepared a script in order to easily run all models, create their predictions and calculate all relevant metrics.

This script should come included in the base level of the zip archives.
Once you have downloaded the data, zip and setup the directories as shown above, please `cd` into the models directory.
There, you only need to run::

   ./reproduce_numbers.sh

And after a few moments, the numbers in the table above should be shown in the console.
