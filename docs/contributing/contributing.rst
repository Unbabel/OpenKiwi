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

Install Poetry via the recommended way::

   curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

It's also possible to use pip::

   pip install poetry

The usual practice of using a virtual environment still applies, possibly also by installing
a local Python interpreter through `pyenv <https://github.com/pyenv/pyenv>`_.

If you don't have Python 3.5, 3.6, or 3.7, or need help creating a virtualenv, check online guides
like `this <https://realpython.com/python-virtual-environments-a-primer/>`_.

**Note**: There's currently an issue with poetry not detecting conda virtual environments. As such
we suggest using virtualenv.

After cloning this repository and creating and activating a virtualenv, instead of the traditional
`pip install -r requirements.txt`, just run::

   poetry install

to install all dependencies.

Then add the package directory to your `PYTHONPATH` to be able to run it::

   export PYTHONPATH=$PYTHONPATH:$(pwd)

Running::

   python kiwi

should now show you a help message.

.. mdinclude:: ../../CONTRIBUTING.md
