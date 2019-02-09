# Contribution guide

## Overview

OpenKiwi is a Quality Estimation toolkit aiming at implementing state of the
art models in an efficient and unified fashion. Since we would like the project
to be open sourced eventually, it is necessary that we follow basic guidelines
in order to ease development, collaboration and readability.

## Basic guidelines

* The project must fully support Python 3.5 or further.
* Code is linted with [flake8](http://flake8.pycqa.org/en/latest/user/error-codes.html), please run `make lint` and fix remaining errors before pushing any code.
* Filenames must be in lowercase.
* Tests are running with [pytest](https://docs.pytest.org/en/latest/) which is commonly referred to the best unittesting framework out there. Pytest implements a standard test discovery which means that it will only search for `test_*.py` or `*_test.py` files. We do not enforce a minimum code coverage but it is preferrable to have even very basic tests running for critical pieces of code. Always test functions that takes/returns tensor argument to document the sizes.
* The `kiwi` folder contains core features. Any script calling these features must be placed into the `scripts` folder.
* Scripts should use [Click](http://click.pocoo.org/5/) instead of `argparse`.

## Contributing

* Keep track of everything by creating issues and editing them with reference to the code! Explain succinctly the problem you are trying to solve and your solution.
* Do not work on `master` directly but in a feature branch, even for small fixes. Always start from an up-to-date `master` branch.
* Work in a clean environment (`virtualenv` is nice). 
* Work within the virtualenv created by Pipenv (`pipenv shell`). For critical, fast changing packages (PyTorch, Tensorflow, ...), set the version in `Pipfile` and update it regularly if it doesn't introduce breaking changes.
* Your commit message must start with an infinitive verb (Add, Fix, Remove, ...).
* Do not merge your code in `master` if your code has not been reviewed before. Always mention team members to add them as reviewers. A small change in appearance can have a big impact in another piece of code you don't know.
* If your change is based on a paper, please include a clear comment and reference in the code and in the related issue.
* In order to test your local changes, install OpenKiwi with `python setup.py develop`.
