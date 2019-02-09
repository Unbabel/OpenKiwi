import pytest

from conftest import check_computation
from kiwi import constants
from kiwi.models.linear_word_qe_classifier import LinearWordQEClassifier


def test_train_linear(temp_output_dir, train_opts, linear_opts, atol):
    train_opts.epochs = 2
    check_computation(
        LinearWordQEClassifier,
        temp_output_dir,
        train_opts,
        linear_opts,
        output_name=constants.TARGET_TAGS,
        expected_avg_probs=0.0,
        atol=atol,
    )


# def test_jackknifing_linear(train_opts, linear_opts, atol):
#     print('Testing jackknifing_linear...')
#     train_dataset, dev_dataset = build_training_datasets(options)
#     splits = 2
#     train_predictions = defaultdict(list)
#     for train_fold, dev_fold in cross_split_dataset(train_dataset, splits):
#         set_train_mode()
#         trainer = build_trainer(options, train_fold, dev_fold)
#         trainer.run()
#         set_predict_mode()
#         predicter = build_predicter(options, dev_fold)
#         predictions = predicter.run()
#         for key, values in predictions.items():
#             train_predictions[key] += values
#     save_predicted_probabilities(options.output_dir, train_predictions)
#     train_predictions = train_predictions[constants.TARGET_TAGS]
#     avg_of_avgs = np.mean(list(map(np.mean, train_predictions)))
#     max_prob = max(map(max, train_predictions))
#     min_prob = min(map(min, train_predictions))
#     np.testing.assert_allclose(avg_of_avgs, 0.485387, atol=atol)
#     assert (0 <= min_prob <= max_prob <= 1)


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
