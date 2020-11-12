
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0](https://github.com/Unbabel/OpenKiwi/compare/2.0.0...2.1.0) - 2020-11-12

### Added
- Hyperparameter search pipeline `kiwi search` built on [Optuna](https://optuna.readthedocs.io/)
- Docs for the search pipeline
- The `--example` flag that has and example config from `kiwi/assests/conf/` printed to terminal for each pipeline
- Tests to increase coverage
- Readme link to the new [OpenKiwiTasting](https://github.com/Unbabel/OpenKiwiTasting) demo.

### Changed
- Example configs in `conf/` so that they are clean, consistent, and have good defaults
- Moved function `feedforward` from `kiwi.tensors` to `kiwi.modules.common.feedforward` where it makes more sense

### Fixed
- The broken relative links in the docs
- Evaluation pipeline by adding missing `quiet` and `verbose` in the evaluate configuration

### Deprecated
- Migration of models from a previous OpenKiwi version, by removing the (never fully working) code in `kiwi.utils.migrations` entirely

### Removed
- Unused code in `kiwi.training.optimizers`, `kiwi.modules.common.scorer`, `kiwi.modules.common.layer_norm`, `kiwi.modules.sentence_level_output`, `kiwi.metrics.metrics`, `kiwi.modules.common.attention`, `kiwi.modules.token_embeddings`
- _All_ code that was already commented out
- The `systems.encoder.(predictor|bert|xlm|xlmrobera).encode_source` option that is both _confusing_ as well as _never used_

## [2.0.0](https://github.com/Unbabel/OpenKiwi/compare/0.1.3...2.0.0)

### Added
- XLMR, XLM, BERT encoder models
- New pooling methods for xlmr-encoder [mixed, mean, ll_mean]
- `freeze_for_number_of_steps` allows freezing of xlmr-encoder for a specific number of training steps
- `encoder_learning_rate` allows to set a specific learning rate to be used on the encoder (different from the rest of the system)
- Dataloaders now use a RandomBucketSampler which groups sentences of the same size together to minimize padding
- fp16 support
- Support for HuggingFace's transformers models
- Pytorch-Lightning as a training framework
- This changelog
