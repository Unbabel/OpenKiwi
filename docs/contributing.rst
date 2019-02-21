Contributing to OpenKiwi
========================

Installing
----------

Please note that since Python>=3.5 is required, all the below commands, especially `pip`,
also have to be the Python 3 version. This might require that you run `pip3` instead.

This project uses a newer configuration format defined by PEP-518, namely, a `pyproject.toml` file.
In order to support it, we use `Poetry <https://github.com/sdispater/poetry>`_ as the build system
and the dependency manager.

Since we want to allow OpenKiwi to be used both as an application and as a library,
this has the added benefit of allowing us to specify dependencies in a single location
and simplifying the packaging process. 
Consequently, you'll notice there's no `requirements.txt` and no `setup.py` files.
The alternative routes are explained below.


For Local Development
---------------------

Before installing OpenKiwi and its development dependencies, we recommend the usual practice of 
creating a separate virtual environment to work on OpenKiwi.

If you don't have Python 3.5, 3.6, or 3.7, or need help creating a virtualenv, check online guides
like `this <https://realpython.com/python-virtual-environments-a-primer/>`_.

**Note**: There's currently an issue with poetry not detecting conda virtual environments. As such
we suggest using virtualenv.

Install Poetry via the recommended way::

   curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

It's also possible to use pip::

   pip install poetry

Then, after cloning this repository and once *inside* your virtualenv, instead of the traditional
`pip install -r requirements.txt`, just run::

   poetry install

to install all dependencies.

That's it! Now, running::

   python kiwi -h

or::
   
   kiwi -h

should show you a help message.

**Optionally** you can use mlflow to track experiments and results. If you'd like to take advantage of
 our integration, simply install Mlflow on the same virtualenv as OpenKiwi.

 You can add it to the poetry environment by running::
   
   poetry install -e mlflow

.. mdinclude:: ../CONTRIBUTING.md
