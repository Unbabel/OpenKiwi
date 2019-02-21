# OpenKiwi

[![PyPI version](https://badge.fury.io/py/openkiwi.svg)](https://badge.fury.io/py/openkiwi)
[![python versions](https://img.shields.io/pypi/pyversions/openkiwi.svg)](https://pypi.org/project/openkiwi/)
[![pipeline status](https://gitlab.com/Unbabel/OpenKiwi/badges/master/pipeline.svg)](https://gitlab.com/Unbabel/OpenKiwi/commits/master)
[![coverage report](https://gitlab.com/Unbabel/OpenKiwi/badges/master/coverage.svg)](https://gitlab.com/Unbabel/OpenKiwi/commits/master)
    
    
Toolkit for Machine Translation Quality Estimation.

Quality estimation(QE) is one of the missing pieces of machine translation: its goal is to evaluate a translation system’s quality without access to reference translations. We present **OpenKiwi**, a Pytorch-based open-source framework that implements the best QE systems from WMT 2015-18 shared tasks, making it easy to experiment with these models under the same framework. Using OpenKiwi and a stacked combination of these models we have achieved state-of-the-art results on the widely used WMT18 English-German dataset.


## Features

* Framework for training QE models and using pre-trained models for evaluating MT
* Supports both word and sentence-level Quality estimation
* Implementation of five QE systems in Pytorch: QUETCH [[1]], NuQE [2, 3], predictor-estimator [4, 5], APE-QE [3], and a stacked ensemble with a linear system [2, 3] 
* Easy to use API. Import it as a package in other projects or run from the command line
* Provides scripts to run pre-trained QE models on data from the WMT 2018 campaign.
* Easy to track and reproduce experiments via yaml configuration files.

[1][Kreutzer et al. (2015): QUality Estimation from ScraTCH (QUETCH): Deep Learning for Word-level Translation Quality Estimation](http://aclweb.org/anthology/W15-3037)

[2][Martins et al. (2016): Unbabel's Participation in the WMT16 Word-Level Translation Quality Estimation Shared Task](http://www.aclweb.org/anthology/W16-2387)

[3][Martins et al. (2017): Pushing the Limits of Translation Quality Estimation](http://www.aclweb.org/anthology/Q17-1015)

[4][Kim et al. (2017): Predictor-Estimator using Multilevel Task Learning with Stack Propagation for Neural Quality Estimation](http://www.aclweb.org/anthology/W17-4763)

[5][Wang et al. (2018): Alibaba Submission for WMT18 Quality Estimation Task](http://statmt.org/wmt18/pdf/WMT093.pdf)


### Using OpenKiwi

To install OpenKiwi as a package, simply run
```bash
pip install openkiwi
```

You can now
```python
import kiwi
```
inside your project or run in the command line
```bash
kiwi
```

**Optionally**, after installation if you'd like to take advantage of our Mlflow integration simply install it in the same virtualenv as OpenKiwi.

Running for example:

```bash
pip install mlflow
```


## Getting Started


Detailed usage examples and instructions can be seen in the [Full Documentation](https://unbabel.github.io/OpenKiwi/index.html).


## Pre-trained models

We provide pre-trained models with the corresponding pre-processed datasets and configuration files. You can easily reproduce our numbers in the WMT18 word-level task by following the instructions available [here](https://unbabel.github.io/OpenKiwi/reproduce.html)

## Contribution

We welcome contributions to improve OpenKiwi. Please refer to [contributing](CONTRIBUTING.md) for quick instructions or to [contributing instructions](https://unbabel.github.io/OpenKiwi/contributing/contributing.html) for more detailed instructions on how to set up your development environment.

## License

OpenKiwi is Affero GPL Licensed. You can see the details of this license in [LICENSE](LICENSE.md).
