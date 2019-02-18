# OpenKiwi

[![pipeline status](https://gitlab.com/Unbabel/OpenKiwi/badges/master/pipeline.svg)](https://gitlab.com/Unbabel/OpenKiwi/commits/master)
[![coverage report](https://gitlab.com/Unbabel/OpenKiwi/badges/master/coverage.svg)](https://gitlab.com/Unbabel/OpenKiwi/commits/master)

Toolkit for Machine Translation Quality Estimation.

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

## Results

http://www.statmt.org/wmt18/quality-estimation-task.html#task2_results

All results are over the respective test set.

### English-Czech (SMT)

Words in MT

Model                  | xF1    | F1_OK  | F1_BAD
-----------------------|--------|--------|-------
NuQE                   | 0.4909 | 0.8208 | 0.5980
Conv64 (Best in WMT18) | 0.4502 | 0.8000 | 0.5628


Gaps in MT

Model                     | xF1    | F1_OK  | F1_BAD
--------------------------|--------|--------|-------
NuQE                      | 0.2076 | 0.9766 | 0.2126
SHEF-bRNN (Best in WMT18) | 0.1740 | 0.9719 | 0.1790


Words in source

Model                     | xF1    | F1_OK  | F1_BAD
--------------------------|--------|--------|-------
NuQE                      | 0.4315 | 0.8250 | 0.5231
SHEF-bRNN (Best in WMT18) | 0.3975 | 0.8114 | 0.4900


### English-Latvian (SMT)

Words in MT

Model                     | xF1    | F1_OK  | F1_BAD
--------------------------|--------|--------|-------
NuQE                      | 0.3830 | 0.8685 | 0.4409
SHEF-PT (Best in WMT18)   | 0.3608 | 0.8685 | 0.4155


Gaps in MT

Model                     | xF1    | F1_OK  | F1_BAD
--------------------------|--------|--------|-------
NuQE                      | 0.1693 | 0.9738 | 0.1739
SHEF-PT (Best in WMT18)   | 0.1364 | 0.9679 | 0.1409


Source in MT

Model                     | xF1    | F1_OK  | F1_BAD
--------------------------|--------|--------|-------
NuQE                      | 0.3082 | 0.8637 | 0.3569
SHEF-bRNN (Best in WMT18) | 0.3057 | 0.8566 | 0.3569


### English-Latvian (NMT)

Words in MT

Model                    | xF1    | F1_OK  | F1_BAD
-------------------------|--------|--------|-------
NuQE                     | 0.4519 | 0.8391 | 0.5386
Conv64 (Best in WMT18)   | 0.4293 | 0.8268 | 0.5192


Gaps in MT

Model                     | xF1    | F1_OK  | F1_BAD
--------------------------|--------|--------|-------
NuQE                      | 0.1289 | 0.9613 | 0.1341
SHEF-PT (Best in WMT18)   | 0.1258 | 0.9653 | 0.1303


Source in MT

Model                   | xF1    | F1_OK  | F1_BAD
------------------------|--------|--------|-------
NuQE                    | 0.3888 | 0.8354 | 0.4654
SHEF-PT (Best in WMT18) | 0.3614 | 0.8137 | 0.4442



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

## Getting Started

Detailed usage examples and instructions can be seen in the full documentation. 
TODO: Insert link to documentation

## Pre-trained models

We provide pre-trained models with the corresponding pre-processed datasets and configuration
files. These can be seen in TODO: INSERT link to location

## Contribution

We welcome contributions to improve OpenKiwi. Please refer to [contributing](CONTRIBUTIN.md) for quick instructions or to TODO: Insert link to documentation for more detailed instructions on how to set up your development environment.

## License

OpenKiwi is Affero GPL Licensed. You can see the details of this license in [LICENSE](LICENSE.md).
