"""
Unit tests for the mathematical core of the network.

Covers the activations, the loss functions, the forward pass, and
backpropagation (against a numerical gradient), plus the weight
initializer. These are the parts an evaluator asks about, and the parts
that silently produced wrong numbers before.
"""

import unittest
import warnings
import numpy as np

from ft_mlp.network_layers import (sigmoid, sigmoid_derivative, softmax,
                                   hidden_layer)
from ft_mlp.loss_functions import (categorical_cross_entropy,
                                   binary_cross_entropy)
from ft_mlp.initializer import he_initialisation
from ft_mlp.model_utils import calculate_accuracy, calculate_loss_mean
from ft_mlp.train import feed_forward, backpropagation, update_weights


def build_model(seed=42, features=5, shapes=(4, 3), outputs=2):
    """Build a small initialized network for testing"""
    model = {
        'input': {'shape': features},
        'layers': [{'shape': n,
                    'activation': sigmoid,
                    'derivative': sigmoid_derivative,
                    'weights_initializer': he_initialisation}
                   for n in shapes],
        'output': {'shape': outputs,
                   'activation': softmax,
                   'weights_initializer': he_initialisation},
        'alpha': 0.1,
    }
    for i, layer in enumerate(model['layers']):
        fan_in = features if i == 0 else model['layers'][i - 1]['shape']
        layer['weights'], layer['bias'] = he_initialisation(
                fan_in, layer['shape'], seed + i)
    model['output']['weights'], model['output']['bias'] = he_initialisation(
            model['layers'][-1]['shape'], outputs, seed + len(shapes))
    return model


class TestActivations(unittest.TestCase):
    """Test activation functions"""

    def test_sigmoid_known_values(self):
        self.assertAlmostEqual(sigmoid(np.array([0.0]))[0], 0.5)
        self.assertAlmostEqual(sigmoid(np.array([1.0]))[0], 0.7310585786300049)
        self.assertAlmostEqual(sigmoid(np.array([-1.0]))[0],
                               0.2689414213699951)

    def test_sigmoid_is_bounded_and_monotonic(self):
        values = np.linspace(-50, 50, 200)
        result = sigmoid(values)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
        self.assertTrue(np.all(np.diff(result) >= 0))

    def test_sigmoid_extreme_values_saturate(self):
        result = sigmoid(np.array([-1000.0, 1000.0]))
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 1.0)
        self.assertFalse(np.any(np.isnan(result)))

    def test_sigmoid_extreme_values_emit_no_warning(self):
        """Overflow warnings during a forward pass look like a broken model"""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = sigmoid(np.array([-1000.0, 1000.0]))
        self.assertFalse(np.any(np.isnan(result)))

    def test_sigmoid_derivative(self):
        # d/dx sigmoid = s * (1 - s), computed from the activation value
        self.assertAlmostEqual(sigmoid_derivative(np.array([0.5]))[0], 0.25)
        self.assertAlmostEqual(sigmoid_derivative(np.array([1.0]))[0], 0.0)

    def test_softmax_rows_sum_to_one(self):
        values = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
        result = softmax(values)
        np.testing.assert_allclose(np.sum(result, axis=1), [1.0, 1.0])
        self.assertTrue(np.all(result > 0))

    def test_softmax_is_numerically_stable(self):
        """Large logits must not overflow: per-row max subtraction"""
        result = softmax(np.array([[1000.0, 1000.0], [0.0, 1000.0]]))
        self.assertFalse(np.any(np.isnan(result)))
        np.testing.assert_allclose(result[0], [0.5, 0.5])
        np.testing.assert_allclose(result[1], [0.0, 1.0], atol=1e-12)

    def test_softmax_shift_invariance(self):
        values = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(softmax(values), softmax(values + 100.0))


class TestLossFunctions(unittest.TestCase):
    """Test loss functions"""

    def test_categorical_cross_entropy_known_value(self):
        prediction = np.array([[0.5, 0.5]])
        truth = np.array([[1, 0]])
        # -log(0.5) = ln 2
        self.assertAlmostEqual(categorical_cross_entropy(prediction, truth)[0],
                               np.log(2), places=6)

    def test_categorical_cross_entropy_perfect_prediction_is_zero(self):
        prediction = np.array([[1.0, 0.0]])
        truth = np.array([[1, 0]])
        self.assertAlmostEqual(categorical_cross_entropy(prediction, truth)[0],
                               0.0, places=6)

    def test_categorical_cross_entropy_returns_one_value_per_sample(self):
        prediction = np.array([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
        truth = np.array([[1, 0], [1, 0], [0, 1]])
        self.assertEqual(categorical_cross_entropy(prediction, truth).shape,
                         (3,))

    def test_categorical_cross_entropy_clips_zero(self):
        """log(0) must not produce -inf"""
        prediction = np.array([[0.0, 1.0]])
        truth = np.array([[1, 0]])
        result = categorical_cross_entropy(prediction, truth)[0]
        self.assertTrue(np.isfinite(result))

    def test_binary_cross_entropy_matches_scalar_form(self):
        """The two-column form must reduce to the subject's scalar formula"""
        prediction = np.array([[0.7, 0.3], [0.2, 0.8]])
        truth = np.array([[1, 0], [0, 1]])
        result = binary_cross_entropy(prediction, truth)
        expected = np.array([-np.log(0.7), -np.log(0.8)])
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestFeedForward(unittest.TestCase):
    """Test the forward pass"""

    def test_feed_forward_shapes(self):
        model = build_model(features=5, shapes=(4, 3), outputs=2)
        inputs = np.random.default_rng(1).normal(size=(7, 5))
        results = feed_forward(model, inputs)
        self.assertEqual(len(results), 3)          # 2 hidden + 1 output
        self.assertEqual(results[0].shape, (7, 4))
        self.assertEqual(results[1].shape, (7, 3))
        self.assertEqual(results[2].shape, (7, 2))

    def test_feed_forward_output_is_a_probability_distribution(self):
        model = build_model()
        inputs = np.random.default_rng(2).normal(size=(6, 5))
        output = feed_forward(model, inputs)[-1]
        np.testing.assert_allclose(np.sum(output, axis=1), np.ones(6))

    def test_hidden_layer_matches_manual_computation(self):
        inputs = np.array([[1.0, 2.0]])
        weights = np.array([[0.5, -0.5], [1.0, 0.25]])
        bias = np.array([0.1, -0.1])
        expected = sigmoid(inputs @ weights + bias)
        result = hidden_layer(inputs, weights, bias, activation=sigmoid)
        np.testing.assert_allclose(result, expected)


class TestBackpropagation(unittest.TestCase):
    """Test backpropagation against a numerical gradient"""

    def _numeric_gradient(self, model, inputs, truth, weights, i, j):
        eps = 1e-6

        def mean_loss():
            out = feed_forward(model, inputs)[-1]
            return float(np.mean(categorical_cross_entropy(out, truth)))

        original = weights[i, j]
        weights[i, j] = original + eps
        plus = mean_loss()
        weights[i, j] = original - eps
        minus = mean_loss()
        weights[i, j] = original
        return (plus - minus) / (2 * eps)

    def test_gradients_match_numerical_gradient_of_mean_loss(self):
        """Backprop must match the gradient of the MEAN, not the SUM

        Matching the sum makes the effective learning rate scale with the
        batch size.
        """
        rng = np.random.default_rng(7)
        batch = 7
        model = build_model(features=5, shapes=(4, 3), outputs=2)
        inputs = rng.normal(size=(batch, 5))
        truth = np.zeros((batch, 2), dtype=int)
        truth[np.arange(batch), rng.integers(0, 2, size=batch)] = 1

        results = feed_forward(model, inputs)
        gradients = backpropagation(model, inputs, results, truth)

        layers = model['layers'] + [model['output']]
        for index, layer in enumerate(layers):
            weights = layer['weights']
            analytic = gradients[index][0]
            self.assertEqual(analytic.shape, weights.shape)
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    numeric = self._numeric_gradient(model, inputs, truth,
                                                     weights, i, j)
                    self.assertAlmostEqual(
                            analytic[i, j], numeric, places=6,
                            msg=f"layer {index} weight ({i},{j})")

    def test_gradient_is_independent_of_batch_size(self):
        """Averaged gradients keep the same scale as the batch grows"""
        rng = np.random.default_rng(11)
        inputs = rng.normal(size=(4, 5))
        truth = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        model = build_model()
        single = backpropagation(model, inputs[:1],
                                 feed_forward(model, inputs[:1]), truth[:1])
        # The same sample repeated 4 times must give the same mean gradient
        repeated_inputs = np.repeat(inputs[:1], 4, axis=0)
        repeated_truth = np.repeat(truth[:1], 4, axis=0)
        repeated = backpropagation(model, repeated_inputs,
                                   feed_forward(model, repeated_inputs),
                                   repeated_truth)
        np.testing.assert_allclose(single[0][0], repeated[0][0], rtol=1e-9)

    def test_gradient_bias_shapes(self):
        model = build_model(features=5, shapes=(4, 3), outputs=2)
        inputs = np.random.default_rng(3).normal(size=(6, 5))
        truth = np.tile(np.array([[1, 0]]), (6, 1))
        gradients = backpropagation(model, inputs,
                                    feed_forward(model, inputs), truth)
        self.assertEqual(gradients[0][1].shape, (4,))
        self.assertEqual(gradients[1][1].shape, (3,))
        self.assertEqual(gradients[2][1].shape, (2,))

    def test_update_weights_decreases_loss(self):
        rng = np.random.default_rng(5)
        model = build_model()
        inputs = rng.normal(size=(10, 5))
        truth = np.zeros((10, 2), dtype=int)
        truth[np.arange(10), rng.integers(0, 2, size=10)] = 1
        model['alpha'] = 0.5

        def loss():
            return calculate_loss_mean(feed_forward(model, inputs)[-1], truth,
                                       categorical_cross_entropy)

        before = loss()
        for _ in range(50):
            results = feed_forward(model, inputs)
            update_weights(model, backpropagation(model, inputs, results,
                                                  truth))
        self.assertLess(loss(), before)


class TestInitializer(unittest.TestCase):
    """Test He initialization"""

    def test_he_uniform_respects_fan_in_bounds(self):
        """heUniform draws from +/- sqrt(6 / fan_in), fan_in = features"""
        fan_in = 30
        weights, _ = he_initialisation(fan_in, 24, 42)
        limit = np.sqrt(6 / fan_in)
        self.assertTrue(np.all(np.abs(weights) <= limit))
        # a uniform draw should get reasonably close to its bounds
        self.assertGreater(np.abs(weights).max(), limit * 0.9)

    def test_he_scale_depends_on_fan_in_not_sample_count(self):
        narrow, _ = he_initialisation(4, 8, 42)
        wide, _ = he_initialisation(400, 8, 42)
        self.assertGreater(np.abs(narrow).max(), np.abs(wide).max())

    def test_he_shapes_and_zero_bias(self):
        weights, bias = he_initialisation(6, 3, 42)
        self.assertEqual(weights.shape, (6, 3))
        np.testing.assert_array_equal(bias, np.zeros(3))

    def test_he_is_reproducible_for_a_given_seed(self):
        first, _ = he_initialisation(5, 5, 42)
        second, _ = he_initialisation(5, 5, 42)
        np.testing.assert_array_equal(first, second)

    def test_he_differs_between_seeds(self):
        """Layers of the same shape must not be initialized identically"""
        first, _ = he_initialisation(24, 24, 42)
        second, _ = he_initialisation(24, 24, 43)
        self.assertFalse(np.array_equal(first, second))


class TestAccuracy(unittest.TestCase):
    """Test the accuracy metric"""

    def test_calculate_accuracy_all_correct(self):
        predictions = np.array([[0.9, 0.1], [0.2, 0.8]])
        truth = np.array([[1, 0], [0, 1]])
        self.assertEqual(calculate_accuracy(predictions, truth), 1.0)

    def test_calculate_accuracy_all_wrong(self):
        predictions = np.array([[0.9, 0.1], [0.2, 0.8]])
        truth = np.array([[0, 1], [1, 0]])
        self.assertEqual(calculate_accuracy(predictions, truth), 0.0)

    def test_calculate_accuracy_half(self):
        predictions = np.array([[0.9, 0.1], [0.2, 0.8]])
        truth = np.array([[1, 0], [1, 0]])
        self.assertEqual(calculate_accuracy(predictions, truth), 0.5)

    def test_calculate_loss_mean_is_the_mean_per_sample_loss(self):
        predictions = np.array([[0.5, 0.5], [0.5, 0.5]])
        truth = np.array([[1, 0], [0, 1]])
        result = calculate_loss_mean(predictions, truth,
                                     categorical_cross_entropy)
        self.assertAlmostEqual(result, np.log(2), places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
