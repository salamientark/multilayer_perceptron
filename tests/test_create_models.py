import argparse as ap
from ft_mlp.create_model import create_model
import unittest

default_valid_model = {
    'epochs': 100,
    'alpha': 0.01,
    'batch': 32,
    'loss':None,
    'seed': 42,
    'optimizer': 'mini-batch',
    'features': [
        'radius_mean',
        'texture_mean',
        'perimeter_mean',
        'area_mean',
        'smoothness_mean',
        'compactness_mean',
        'concavity_mean',
        'concave_points_mean',
        'symmetry_mean',
        'fractal_dimension_mean',
        'radius_std',
        'texture_std',
        'perimeter_std',
        'area_std',
        'smoothness_std',
        'compactness_std',
        'concavity_std',
        'concave_points_std',
        'symmetry_std',
        'fractal_dimension_std',
        'radius_worst',
        'texture_worst',
        'perimeter_worst',
        'area_worst',
        'smoothness_worst',
        'compactness_worst',
        'concavity_worst',
        'concave_points_worst',
        'symmetry_worst',
        'fractal_dimension_worst'
    ],
    'target': 'diagnosis',
    'input': {
        'shape': 30,
        'activation': 'sigmoid',
        'weights_initializer': 'heUniform'
    },
    'layers': [
        {
            'shape': 64,
            'activation': 'sigmoid',
            'weights_initializer': 'heUniform'
        },
        {
            'shape': 32,
            'activation': 'sigmoid',
            'weights_initializer': 'heUniform'
        },
        {
            'shape': 16,
            'activation': 'sigmoid',
            'weights_initializer': 'heUniform'
        }
    ],
    'output': {
        'shape': 2,
        'activation': 'softmax',
        'weights_initializer': 'heUniform'
    },
}


class test_create_models(unittest.TestCase):
    def test_create_model_no_features(self):
        args = ap.Namespace(
            conf=None,
            epochs=100,
            alpha=0.001,
            train_ratio=None,
            batch=32,
            loss='categoricalCrossentropy',
            seed=42,
            optimizer='mini-batch',
            shape=[128, 64, 32],
            features=None,
            dataset='tests/test_data/data.csv'
        )
        target = 'diagnosis'
        result = create_model(args, target)
        self.assertEqual(result, default_valid_model)  # Placeholder assertion

    # def test_create_product_model(self):
    #     # Test code for creating product model
    #     self.assertTrue(True)  # Placeholder assertion
