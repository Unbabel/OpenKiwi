
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
