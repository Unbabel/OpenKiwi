# Contribution guide

## Overview

OpenKiwi is an Open Source Quality Estimation toolkit aimed at implementing state of the art models in an efficient and unified fashion. While we do welcome contributions, in order to guarantee their quality and usefulness, it is necessary that we follow basic guidelines in order to ease development, collaboration and readability.

## Basic guidelines

* The project must fully support Python 3.5 or further.
* Code is linted with [flake8](http://flake8.pycqa.org/en/latest/user/error-codes.html), please run `flake8 kiwi` and fix remaining errors before pushing any code.
* Code formatting must stick to the Facebook style, 80 columns and single quotes. For Python 3.6+, the [black](https://github.com/ambv/black) formatter can be used by running `Black kiwi`. For python 3.5, [YAPF](https://github.com/google/yapf) should get most of the job done, although some manual changes might be necessary.
* Imports are sorted with [isort](https://github.com/timothycrosley/isort).
* Filenames must be in lowercase.
* Tests are running with [pytest](https://docs.pytest.org/en/latest/) which is commonly referred to the best unittesting framework out there. Pytest implements a standard test discovery which means that it will only search for `test_*.py` or `*_test.py` files. We do not enforce a minimum code coverage but it is preferrable to have even very basic tests running for critical pieces of code. Always test functions that takes/returns tensor argument to document the sizes.
* The `kiwi` folder contains core features. Any script calling these features must be placed into the `scripts` folder.

## Contributing

* Keep track of everything by creating issues and editing them with reference to the code! Explain succinctly the problem you are trying to solve and your solution.
* Contributions to `master` should be made through github pull-requests.
* Dependencies are managed using `Poetry`. Although we would rather err on the side of less rather than more dependencies, if needed they are managed through the `pyproject.toml` file.
* Work in a clean environment (`virtualenv` is nice). 
* Your commit message must start with an infinitive verb (Add, Fix, Remove, ...).
* If your change is based on a paper, please include a clear comment and reference in the code and in the related issue.
* In order to test your local changes, install OpenKiwi following the instructions on the [documentation](https://unbabel.github.io/openkiwi)
