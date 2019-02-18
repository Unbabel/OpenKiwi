import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kiwi.lib import jackknife, predict, train
from kiwi.lib.utils import configure_seed
from kiwi.loggers import mlflow_logger


@pytest.fixture(scope='function')
def temp_output_dir():
    output_dir = tempfile.mkdtemp()
    yield output_dir
    shutil.rmtree(output_dir)


@pytest.fixture(scope='session')
def atol():
    return 1e-2


@pytest.fixture(scope='session')
def base_dir():
    return Path(__file__).absolute().parent.joinpath('./toy-data/')


@pytest.fixture(scope='session')
def dir_17(base_dir):
    return base_dir.joinpath('WMT17/word_level/en_de/')


@pytest.fixture(scope='session')
def dir_18(base_dir):
    return base_dir.joinpath('WMT18/')


@pytest.fixture(scope='session')
def predest_dir(base_dir):
    return base_dir.joinpath('predest/')


@pytest.fixture
def data_opts_17(dir_17):
    data_opts = argparse.Namespace()
    data_opts.wmt18_format = False
    data_opts.test_source = str(dir_17.joinpath('dev.src'))
    # data_opts.test_source_pos = str(dir_17.joinpath('dev.src.pos'))
    data_opts.test_source_tags = None
    data_opts.test_target = str(dir_17.joinpath('dev.mt'))
    # data_opts.test_target_pos = str(dir_17.joinpath('dev.mt.pos'))
    data_opts.test_target_tags = str(dir_17.joinpath('dev.tags'))
    data_opts.test_alignments = str(dir_17.joinpath('dev.align'))

    data_opts.train_source = str(dir_17.joinpath('train.src'))
    # data_opts.train_source_pos = str(dir_17.joinpath('train.src.pos'))
    data_opts.train_source_tags = None
    data_opts.train_target = str(dir_17.joinpath('train.mt'))
    # data_opts.train_target_pos = str(dir_17.joinpath('train.mt.pos'))
    data_opts.train_target_tags = str(dir_17.joinpath('train.tags'))
    data_opts.train_alignments = str(dir_17.joinpath('train.align'))

    data_opts.valid_source = str(dir_17.joinpath('dev.src'))
    # data_opts.valid_source_pos = str(dir_17.joinpath('dev.src.pos'))
    data_opts.valid_source_tags = None
    data_opts.valid_target = str(dir_17.joinpath('dev.mt'))
    # data_opts.valid_target_pos = str(dir_17.joinpath('dev.mt.pos'))
    data_opts.valid_target_tags = str(dir_17.joinpath('dev.tags'))
    data_opts.valid_alignments = str(dir_17.joinpath('dev.align'))

    # pruning
    data_opts.source_max_length = 50
    data_opts.source_min_length = 1
    data_opts.target_max_length = 50
    data_opts.target_min_length = 1

    # vocabulary
    data_opts.source_vocab_size = 100000
    data_opts.target_vocab_min_frequency = 1
    data_opts.target_vocab_size = 100000

    return data_opts


@pytest.fixture
def data_opts_18(data_opts_17, dir_18):
    data_opts = data_opts_17
    data_opts.wmt18_format = True
    data_opts.test_source = str(dir_18.joinpath('dev.src'))
    data_opts.test_source_pos = None
    data_opts.test_source_tags = str(dir_18.joinpath('dev.src_tags'))
    data_opts.test_target = str(dir_18.joinpath('dev.mt'))
    data_opts.test_target_pos = None
    data_opts.test_target_tags = str(dir_18.joinpath('dev.tags'))
    data_opts.test_alignments = str(dir_18.joinpath('dev.src-mt.alignments'))

    data_opts.train_source = str(dir_18.joinpath('train.src'))
    data_opts.train_source_pos = None
    data_opts.train_source_tags = str(dir_18.joinpath('train.src_tags'))
    data_opts.train_target = str(dir_18.joinpath('train.mt'))
    data_opts.train_target_pos = None
    data_opts.train_target_tags = str(dir_18.joinpath('train.tags'))
    data_opts.train_alignments = str(dir_18.joinpath('train.src-mt.alignments'))

    data_opts.valid_source = str(dir_18.joinpath('dev.src'))
    data_opts.valid_source_pos = None
    data_opts.valid_source_tags = str(dir_18.joinpath('dev.src_tags'))
    data_opts.valid_target = str(dir_18.joinpath('dev.mt'))
    data_opts.valid_target_pos = None
    data_opts.valid_target_tags = str(dir_18.joinpath('dev.tags'))
    data_opts.valid_alignments = str(dir_18.joinpath('dev.src-mt.alignments'))

    return data_opts


@pytest.fixture(scope='function')
def general_opts():
    options = argparse.Namespace()
    # general
    # options.output_dir = setup
    options.seed = 42
    configure_seed(options.seed)

    # gpu
    options.gpu_id = None
    options.gpu_verbose_level = 0

    # logging
    options.debug = False
    options.quiet = False
    options.log_interval = 0

    # set save and load to None
    options.save_model = None
    options.load_model = None
    options.save_data = None
    return options


@pytest.fixture(scope='function')
def train_opts(general_opts):
    train_opts = general_opts
    # training
    train_opts.epochs = 2
    train_opts.shuffle = True
    train_opts.train_batch_size = 8
    train_opts.valid_batch_size = 8
    train_opts.batch_size = 8

    train_opts.load_vocab = None
    train_opts.load_data = None
    train_opts.extend_source_vocab = None
    train_opts.extend_target_vocab = None
    # optimizer
    train_opts.optimizer = 'adam'
    train_opts.learning_rate = 0.001
    train_opts.learning_rate_decay = 1.0
    train_opts.learning_rate_decay_start = 0

    # Checkpointing
    # train_opts.checkpoint_validation_epochs = 1
    train_opts.checkpoint_validation_steps = 100
    train_opts.checkpoint_save = True
    train_opts.checkpoint_keep_only_best = 0
    train_opts.checkpoint_early_stop_patience = 0
    train_opts.resume = False
    return train_opts


@pytest.fixture(scope='function')
def predest_opts(data_opts_17):
    model_opts = data_opts_17
    model_opts.load_pred_source = None
    model_opts.load_pred_target = None

    model_opts.extend_source_vocab = None
    model_opts.extend_target_vocab = None

    # network
    model_opts.embedding_sizes = 0
    model_opts.share_embeddings = False
    model_opts.source_embeddings_size = 25
    model_opts.target_embeddings_size = 25
    model_opts.output_size = None
    model_opts.dropout = 0.0
    model_opts.embeddings_dropout = 0.0
    model_opts.target_bad_weight = 5.0
    model_opts.gaps_bad_weight = 5.0
    model_opts.source_bad_weight = 5.0

    # predest
    model_opts.warmup = 0
    model_opts.rnn_layers_pred = 2
    model_opts.rnn_layers_est = 1
    model_opts.dropout_est = 0.0
    model_opts.dropout_pred = 0.0
    model_opts.hidden_pred = 50
    model_opts.hidden_est = 25
    model_opts.mlp_est = False
    model_opts.out_embeddings_size = 25
    model_opts.start_stop = False

    # multilevel
    model_opts.predict_gaps = False
    model_opts.predict_source = False
    model_opts.predict_target = True
    model_opts.predict_inverse = False
    model_opts.token_level = False
    model_opts.sentence_level = False
    model_opts.sentence_ll = False
    model_opts.use_probs = False
    model_opts.binary_level = False

    return model_opts


@pytest.fixture(scope='function')
def predictor_opts(data_opts_17, predest_opts):
    return predest_opts


@pytest.fixture
def quetch_opts(data_opts_17):
    model_opts = data_opts_17

    model_opts.sentence_level = False
    model_opts.predict_target = True
    model_opts.predict_source = False
    model_opts.predict_gaps = False
    model_opts.keep_rare_words_with_embeddings = True
    model_opts.add_embeddings_vocab = False

    # embeddings
    model_opts.source_embeddings = None
    model_opts.target_embeddings = None
    model_opts.embeddings_format = 'polyglot'
    model_opts.embeddings_binary = False

    # network
    model_opts.source_embeddings_size = 50
    model_opts.target_embeddings_size = 50
    model_opts.hidden_sizes = [20]
    model_opts.output_size = None
    model_opts.dropout = 0.0
    model_opts.embeddings_dropout = 0.0
    model_opts.freeze_embeddings = False
    model_opts.bad_weight = 3.0

    # initialization
    model_opts.init_support = 0.1
    model_opts.init_type = 'uniform'

    # nuqe
    model_opts.window_size = 3
    model_opts.max_aligned = 5
    return model_opts


@pytest.fixture
def nuqe_opts(quetch_opts):
    nuqe_opts = quetch_opts
    nuqe_opts.hidden_sizes = [40, 20, 10, 5]
    return nuqe_opts


@pytest.fixture
def linear_opts(data_opts_17, dir_17):
    model_opts = data_opts_17

    model_opts.use_basic_features_only = 0
    model_opts.use_bigrams = 1
    model_opts.use_simple_bigram_features = 0
    model_opts.training_algorithm = 'svm_mira'
    model_opts.regularization_constant = 0.001
    model_opts.cost_false_positives = 0.2
    model_opts.cost_false_negatives = 0.8

    model_opts.evaluation_metric = 'f1_mult'

    model_opts.train_source_pos = str(
        dir_17.joinpath('additional', 'train.src.pos')
    )
    model_opts.train_target_pos = str(
        dir_17.joinpath('additional', 'train.mt.pos')
    )
    model_opts.train_target_ngram = str(
        dir_17.joinpath('features', 'train.mt.ngrams')
    )
    model_opts.dev_source_pos = str(
        dir_17.joinpath('additional', 'dev.src.pos')
    )
    model_opts.dev_target_pos = str(dir_17.joinpath('additional', 'dev.mt.pos'))
    model_opts.dev_target_ngram = str(
        dir_17.joinpath('features', 'dev.mt.ngrams')
    )
    model_opts.dev_target_parse = str(
        dir_17.joinpath('additional', 'dev.mt.parses')
    )
    model_opts.dev_target_stacked = str(
        dir_17.joinpath('predictions', 'dev.nuqe.stacked')
    )
    model_opts.train_target_parse = str(
        dir_17.joinpath('additional', 'train.mt.parses')
    )
    model_opts.train_target_stacked = str(
        dir_17.joinpath('predictions', 'train.nuqe.stacked')
    )

    return model_opts


def check_computation(
    model_api,
    output_dir,
    pipeline_opts,
    model_opts,
    output_name,
    expected_avg_probs,
    atol,
):
    # Testing training
    pipeline_opts.save_model = output_dir
    pipeline_opts.load_model = None

    mlflow_run = mlflow_logger.configure(
        run_uuid=None,
        experiment_name='Tests',
        tracking_uri=str(Path(output_dir, 'mlruns')),
    )
    with mlflow_run:
        trainer = train.run(model_api, output_dir, pipeline_opts, model_opts)

    # Testing prediction
    pipeline_opts.load_model = trainer.checkpointer.best_model_path()

    predictions = predict.run(model_api, output_dir, pipeline_opts, model_opts)
    predictions = predictions[output_name]
    avg_of_avgs = np.mean(list(map(np.mean, predictions)))
    max_prob = max(map(max, predictions))
    min_prob = min(map(min, predictions))
    np.testing.assert_allclose(avg_of_avgs, expected_avg_probs, atol=atol)
    assert 0 <= min_prob <= avg_of_avgs <= max_prob <= 1


def check_jackknife(
    model_api,
    output_dir,
    pipeline_opts,
    model_opts,
    output_name,
    expected_avg_probs,
    atol,
):
    train_opts.save_model = output_dir
    train_opts.load_model = None

    mlflow_run = mlflow_logger.configure(
        run_uuid=None,
        experiment_name='Tests',
        tracking_uri=str(Path(output_dir, 'mlruns')),
    )
    with mlflow_run:
        train_predictions = jackknife.run(
            model_api, output_dir, pipeline_opts, model_opts, splits=2
        )
    train_predictions = train_predictions[output_name]
    avg_of_avgs = np.mean(list(map(np.mean, train_predictions)))
    max_prob = max(map(max, train_predictions))
    min_prob = min(map(min, train_predictions))
    np.testing.assert_allclose(avg_of_avgs, expected_avg_probs, atol=atol)
    assert 0 <= min_prob <= avg_of_avgs <= max_prob <= 1
