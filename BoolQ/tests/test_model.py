# -*- coding: utf-8 -*-
from allennlp.common.testing import ModelTestCase

from boolq import BoolQDatasetReader
from .const import TEST_FIXTURES_PATH


class TestSimpleClassifier(ModelTestCase):
    def test_model_can_train(self):
        # This built-in test makes sure that your data can load, that it gets passed to the model
        # correctly, that your model computes a loss in a way that we can get gradients from it,
        # that all of your parameters get non-zero gradient updates, and that we can save and load
        # your model and have the model's predictions remain consistent.
        param_file = str(TEST_FIXTURES_PATH / "default_config.json")
        self.ensure_model_can_train_save_and_load(param_file)
