"""A generic implementation of a basic tester."""

from kiwi import constants as const


class LinearTester(object):
    def __init__(self, classifier):
        self.classifier = classifier

    def run(self, dataset, **kwargs):
        instances = self.classifier.create_instances(dataset)
        predictions = self.classifier.test(instances)
        self.classifier.evaluate(instances, predictions)
        return {const.TARGET_TAGS: predictions}
