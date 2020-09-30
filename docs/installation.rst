.. _installation:

Installation
============

OpenKiwi can be installed in two ways, depending on how it's going to be used.

As a library
------------

Simply run::

   pip install openkiwi

You can now::

   import kiwi

inside your project or run in the command line::

   kiwi


As a local package
------------------

OpenKiwi's configuration is in ``pyproject.toml`` (as defined by PEP-518).
We use `Poetry <https://github.com/sdispater/poetry>`_ as the build system
and the dependency manager. All dependencies are specified in that file.

Install Poetry via the recommended way::

   curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

It's also possible to use pip (but not recommended as it might mess up local dependencies)::

   pip install poetry

In your virtualenv just run::

   poetry install

to install all dependencies.

That's it! Now, running::

   python kiwi -h

or::

   kiwi -h

should show you a help message.


MLflow integration
------------------

**Optionally**, to take advantage of our `MLflow <https://mlflow.org/>`_ integration, install Kiwi with::

   pip install openkiwi[mlflow]


**Or**::

   poetry install -E mlflow

