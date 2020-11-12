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

A new major version (2.0.0) of OpenKiwi has been released. Introducing HuggingFace Transformers support and adoption of Pytorch-lightning.
For a condensed view of changed, check the [changelog](https://github.com/Unbabel/OpenKiwi/blob/master/CHANGELOG.md)

Following our nomination in early July, we are happy to announce we won the [Best Demo Paper at ACL 2019](http://www.acl2019.org/EN/winners-of-acl-2019-best-paper-awards.xhtml)! Congratulations to the whole team and huge thanks for supporters and issue reporters.

Check out the [published paper](https://www.aclweb.org/anthology/P19-3020).

We have released the OpenKiwi [tutorial](https://github.com/Unbabel/KiwiCutter) we presented at MT Marathon 2019.

## Features

* Framework for training QE models and using pre-trained models for evaluating MT.
* Supports both word and sentence-level (HTER or z-score) Quality estimation.
* Implementation of five QE systems in Pytorch: NuQE [[2], [3]], predictor-estimator [[4], [5]], BERT-Estimator [[6]], XLM-Estimator [[6]] and XLMR-Estimator
*    Older systems only supported in versions <=2.0.0: QUETCH [[1]], APE-QE [[3]] and a stacked ensemble with a linear system [[2], [3]].
* Easy to use API. Import it as a package in other projects or run from the command line.
* Easy to track and reproduce experiments via yaml configuration files.
* Based on Pytorch-Lightning making the code easier to scale, use and keep up-do-date with engineering advances.
* Implemented using HuggingFace Transformers library to allow easy access to state-of-the-art pre-trained models.

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
pip install openkiwi[mlflow]
```


## Getting Started

Detailed usage examples and instructions can be found in the [Full Documentation](https://unbabel.github.io/OpenKiwi/index.html).


## Contributing

We welcome contributions to improve OpenKiwi.
Please refer to [CONTRIBUTING.md](https://github.com/Unbabel/OpenKiwi/blob/master/CONTRIBUTING.md) for quick instructions or to [contributing instructions](https://unbabel.github.io/OpenKiwi/contributing.html) for more detailed instructions on how to set up your development environment.


## License

OpenKiwi is Affero GPL licensed. You can see the details of this license in [LICENSE](https://github.com/Unbabel/OpenKiwi/blob/master/LICENSE).


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

##### [[6]] [Kepler et al. (2019): Unbabel’s Participation in the WMT19 Translation Quality Estimation Shared Task](https://www.aclweb.org/anthology/W19-5406.pdf)
[6]:#6-kepler-et-al-2019-unbabels-participation-in-the-wmt19-translation-quality-estimation-shared-task
