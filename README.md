![OpenKiwi Logo](https://github.com/Unbabel/OpenKiwi/blob/master/docs/_static/img/openkiwi-logo-horizontal.svg)

--------------------------------------------------------------------------------

[![PyPI version](https://img.shields.io/pypi/v/openkiwi?color=%236ecfbd&label=pypi%20package&style=flat-square)](https://pypi.org/project/openkiwi/)
[![python versions](https://img.shields.io/pypi/pyversions/openkiwi.svg?style=flat-square)](https://pypi.org/project/openkiwi/)
[![CircleCI](https://img.shields.io/circleci/build/github/Unbabel/OpenKiwi/master?style=flat-square)](https://circleci.com/gh/Unbabel/OpenKiwi/tree/master)
[![Code Climate coverage](https://img.shields.io/codeclimate/coverage/Unbabel/OpenKiwi?style=flat-square)](https://codeclimate.com/github/Unbabel/OpenKiwi/test_coverage)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![GitHub last commit](https://img.shields.io/github/last-commit/unbabel/openkiwi?style=flat-square)](https://github.com/Unbabel/OpenKiwi/commits/master)

**Open-Source Machine Translation Quality Estimation in PyTorch**

Quality estimation (QE) is one of the missing pieces of machine translation: its goal is to evaluate a translation system’s quality without access to reference translations. We present **OpenKiwi**, a Pytorch-based open-source framework that implements the best QE systems from WMT 2015-18 shared tasks, making it easy to experiment with these models under the same framework. Using OpenKiwi and a stacked combination of these models we have achieved state-of-the-art results on word-level QE on the WMT 2018 English-German dataset.

## News

Following our nomination in early July, we are happy to announce we won the [Best Demo Paper at ACL 2019](http://www.acl2019.org/EN/winners-of-acl-2019-best-paper-awards.xhtml)! Congratulations to the whole team and huge thanks for supporters and issue reporters.

Check out the [published paper](https://www.aclweb.org/anthology/P19-3020).

We are going to release the web interface we had put in place for the live demo presentation at ACL.

We have released the OpenKiwi [tutorial](https://github.com/Unbabel/KiwiCutter) we presented at MT Marathon 2019.

## Features

* Framework for training QE models and using pre-trained models for evaluating MT.
* Supports both word and sentence-level Quality estimation.
* Implementation of five QE systems in Pytorch: QUETCH [[1]], NuQE [[2], [3]], predictor-estimator [[4], [5]], APE-QE [[3]], and a stacked ensemble with a linear system [[2], [3]]. 
* Easy to use API. Import it as a package in other projects or run from the command line.
* Provides scripts to run pre-trained QE models on data from the WMT 2018 campaign.
* Easy to track and reproduce experiments via yaml configuration files.

## Results

Results for the [WMT18 Quality Estimation shared task](http://www.statmt.org/wmt18/quality-estimation-task.html#results), for [word level](https://competitions.codalab.org/competitions/19306#results) *and* [sentence level](https://competitions.codalab.org/competitions/19316#results) on the test set.

|   Model   | En-De SMT |           |           |           |           | En-De NMT |           |           |           |           |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|           |     MT    |    gaps   |   source  |     r     |     ⍴     |     MT    |    gaps   |   source  |     r     |     ⍴     |
|  OpenKiwi | **62.70** | **52.14** | **48.88** |   71.08   |   72.70   | **44.77** | **22.89** | **36.53** |   46.72   |   58.51   |
|  Wang2018 |   62.46   |   49.99   |     --    | **73.97** | **75.43** |   43.61   |     --    |     --    |   50.12   |   60.49   |
|    UNQE   |     --    |     --    |     --    |   70.00   |   72.44   |     --    |     --    |     --    | **51.29** | **60.52** |
| deepQUEST |   42.98   |   28.24   |   33.97   |   48.72   |   50.97   |   30.31   |   11.93   |   28.59   |   38.08   |   48.00   |


## Quick Installation

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

**Optionally**, if you'd like to take advantage of our [MLflow](https://mlflow.org/) integration, simply install it in the same virtualenv as OpenKiwi:
```bash
pip install mlflow
```


## Getting Started

Detailed usage examples and instructions can be found in the [Full Documentation](https://unbabel.github.io/OpenKiwi/index.html).


## Pre-trained models

We provide pre-trained models with the corresponding pre-processed datasets and configuration files.
You can easily reproduce our numbers in the WMT 2018 word- and sentence-level tasks by following the [reproduce instructions in the documentation](https://unbabel.github.io/OpenKiwi/reproduce.html).


## Contributing

We welcome contributions to improve OpenKiwi.
Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for quick instructions or to [contributing instructions](https://unbabel.github.io/OpenKiwi/contributing.html) for more detailed instructions on how to set up your development environment.


## License

OpenKiwi is Affero GPL licensed. You can see the details of this license in [LICENSE](LICENSE).


## Citation

If you use OpenKiwi, please cite the following paper: [OpenKiwi: An Open Source Framework for Quality Estimation](https://www.aclweb.org/anthology/P19-3020).

```
@inproceedings{openkiwi,
    author = {Fábio Kepler and
              Jonay Trénous and
              Marcos Treviso and
              Miguel Vera and
              André F. T. Martins},
    title  = {Open{K}iwi: An Open Source Framework for Quality Estimation},
    year   = {2019},
    booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics--System Demonstrations},
    pages  = {117--122},
    month  = {July},
    address = {Florence, Italy},
    url    = {https://www.aclweb.org/anthology/P19-3020},
    organization = {Association for Computational Linguistics},
}
```


## References

##### [[1]] [Kreutzer et al. (2015): QUality Estimation from ScraTCH (QUETCH): Deep Learning for Word-level Translation Quality Estimation](http://aclweb.org/anthology/W15-3037)
[1]:#1-kreutzer-et-al-2015-quality-estimation-from-scratch-quetch-deep-learning-for-word-level-translation-quality-estimation

##### [[2]] [Martins et al. (2016): Unbabel's Participation in the WMT16 Word-Level Translation Quality Estimation Shared Task](http://www.aclweb.org/anthology/W16-2387)
[2]:#2-martins-et-al-2016-unbabels-participation-in-the-wmt16-word-level-translation-quality-estimation-shared-task

##### [[3]] [Martins et al. (2017): Pushing the Limits of Translation Quality Estimation](http://www.aclweb.org/anthology/Q17-1015)
[3]:#3-martins-et-al-2017-pushing-the-limits-of-translation-quality-estimation

##### [[4]] [Kim et al. (2017): Predictor-Estimator using Multilevel Task Learning with Stack Propagation for Neural Quality Estimation](http://www.aclweb.org/anthology/W17-4763)
[4]:#4-kim-et-al-2017-predictor-estimator-using-multilevel-task-learning-with-stack-propagation-for-neural-quality-estimation

##### [[5]] [Wang et al. (2018): Alibaba Submission for WMT18 Quality Estimation Task](http://statmt.org/wmt18/pdf/WMT093.pdf)
[5]:#5-wang-et-al-2018-alibaba-submission-for-wmt18-quality-estimation-task
