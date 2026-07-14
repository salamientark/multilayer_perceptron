"""
Unit tests for model persistence and metric helpers.
"""

import contextlib
import io
import os
import json
import tempfile
import unittest
import numpy as np

from ft_mlp.model_utils import (print_model, get_random_seed,
                                get_random_batch_indexes, save_weights,
                                load_weights_from_file, save_model,
                                FUNCTION_NAME)
from ft_mlp.network_layers import sigmoid, softmax, predict, score_function
from ft_mlp.loss_functions import categorical_cross_entropy
from ft_mlp.initializer import he_initialisation


def dummy_model():
    weights_a, bias_a = he_initialisation(3, 4, 42)
    weights_b, bias_b = he_initialisation(4, 2, 43)
    return {
        'epoch': 10, 'alpha': 0.1, 'batch': 8,
        'loss': categorical_cross_entropy, 'seed': 42,
        'optimizer': 'mini-batch',
        'features': ['a', 'b', 'c'], 'target': 'diagnosis',
        'classes': ['B', 'M'],
        'standardization': {'a': {'mean': 1.0, 'std': 2.0},
                            'b': {'mean': 0.0, 'std': 1.0},
                            'c': {'mean': -1.0, 'std': 3.0}},
        'data_train': np.zeros((20, 4)),
        'input': {'shape': 3},
        'layers': [{'shape': 4, 'activation': sigmoid,
                    'weights_initializer': he_initialisation,
                    'weights': weights_a, 'bias': bias_a}],
        'output': {'shape': 2, 'activation': softmax,
                   'weights_initializer': he_initialisation,
                   'weights': weights_b, 'bias': bias_b},
        }


class TestRandomHelpers(unittest.TestCase):

    def test_get_random_seed_is_a_positive_int(self):
        for _ in range(5):
            value = get_random_seed()
            self.assertIsInstance(value, int)
            self.assertGreater(value, 0)

    def test_batch_indexes_are_a_permutation(self):
        result = get_random_batch_indexes(10, seed=1)
        self.assertEqual(sorted(result.tolist()), list(range(10)))

    def test_batch_indexes_are_reproducible_with_a_seed(self):
        np.testing.assert_array_equal(get_random_batch_indexes(20, seed=3),
                                      get_random_batch_indexes(20, seed=3))

    def test_batch_indexes_differ_between_seeds(self):
        first = get_random_batch_indexes(50, seed=1)
        second = get_random_batch_indexes(50, seed=2)
        self.assertFalse(np.array_equal(first, second))

    def test_batch_indexes_without_seed_still_permute(self):
        result = get_random_batch_indexes(8)
        self.assertEqual(sorted(result.tolist()), list(range(8)))


class TestPrintModel(unittest.TestCase):

    def test_print_model_renders_functions_by_name(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_model({'loss': sigmoid, 'shape': 3})
        text = output.getvalue()
        self.assertIn('sigmoid', text)


class TestWeightsPersistence(unittest.TestCase):

    def test_save_and_load_weights_round_trip(self):
        model = dummy_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'weights.npz')
            with contextlib.redirect_stdout(io.StringIO()):
                save_weights(path, model)
            loaded = load_weights_from_file(path)
            np.testing.assert_allclose(loaded['layer_0_weights'],
                                       model['layers'][0]['weights'])
            np.testing.assert_allclose(loaded['layer_0_bias'],
                                       model['layers'][0]['bias'])
            np.testing.assert_allclose(loaded['output_weights'],
                                       model['output']['weights'])
            np.testing.assert_allclose(loaded['output_bias'],
                                       model['output']['bias'])


class TestSaveModel(unittest.TestCase):

    def _saved(self, model=None):
        model = model or dummy_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.json')
            with contextlib.redirect_stdout(io.StringIO()):
                save_model(path, model)
            with open(path) as f:
                return json.load(f)

    def test_functions_are_saved_by_name(self):
        saved = self._saved()
        self.assertEqual(saved['loss'], 'categoricalCrossentropy')
        self.assertEqual(saved['output']['activation'], 'softmax')
        self.assertEqual(saved['output']['weights_initializer'], 'heUniform')
        self.assertEqual(saved['layers'][0]['activation'], 'sigmoid')

    def test_topology_is_saved(self):
        saved = self._saved()
        self.assertEqual(saved['input']['shape'], 3)
        self.assertEqual(saved['layers'][0]['shape'], 4)
        self.assertEqual(saved['output']['shape'], 2)

    def test_classes_and_standardization_are_saved(self):
        saved = self._saved()
        self.assertEqual(saved['classes'], ['B', 'M'])
        self.assertAlmostEqual(saved['standardization']['a']['mean'], 1.0)
        self.assertAlmostEqual(saved['standardization']['a']['std'], 2.0)

    def test_batch_is_clamped_to_the_training_set_size(self):
        model = dummy_model()
        model['batch'] = 999          # larger than the 20 training rows
        self.assertEqual(self._saved(model)['batch'], 20)

    def test_hyperparameters_are_saved(self):
        saved = self._saved()
        self.assertEqual(saved['epoch'], 10)
        self.assertEqual(saved['seed'], 42)
        self.assertEqual(saved['optimizer'], 'mini-batch')
        self.assertEqual(saved['features'], ['a', 'b', 'c'])
        self.assertEqual(saved['target'], 'diagnosis')

    def test_function_name_map_is_complete(self):
        for function in (sigmoid, softmax, categorical_cross_entropy,
                         he_initialisation):
            self.assertIn(function, FUNCTION_NAME)


class TestNetworkPredict(unittest.TestCase):

    def test_predict_returns_a_distribution_per_row(self):
        model = dummy_model()
        data = np.random.default_rng(1).normal(size=(5, 3))
        result = predict(model, data)
        self.assertEqual(result.shape, (5, 2))
        np.testing.assert_allclose(np.sum(result, axis=1), np.ones(5))

    def test_score_function_applies_weights_and_bias(self):
        weights = np.array([[1.0, 2.0]])
        features = np.array([[3.0], [4.0]])
        result = score_function(weights, features, bias=np.array([[1.0]]))
        np.testing.assert_allclose(result, [[12.0]])

    def test_score_function_without_bias(self):
        weights = np.array([[1.0, 2.0]])
        features = np.array([[3.0], [4.0]])
        result = score_function(weights, features, bias=None)
        np.testing.assert_allclose(result, [[11.0]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
