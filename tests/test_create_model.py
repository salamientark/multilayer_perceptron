"""
Unit tests for create)model modules in ft_mlp/

"""

# args = {
#     "--shape": [10, 5],
#     "--layer": 3,
#     "--neurons": 8,
#     "--conf": "model_config.json",
#     "--features": "feature_columns.json",
#     "--loss": "categoricalCrossentropy",
#     "--epoch": 100,
#     "--learning_rate": 0.01, -> alpha
#     "--batch": 16,
#     "--seed": 42,
#     "--train_ratio": 0.8,
#     "--outfile": "trained_model.json",
#     "dataset": "data_training.csv",
# }

import unittest
import sys
import io
import argparse
from ft_mlp.create_model import init_model_template, fill_model_from_json, fill_model_from_param, fill_model_datasets, create_model, load_model_from_json
from ft_mlp.network_layers import sigmoid, softmax
from ft_mlp.initializer import he_initialisation
from ft_mlp.loss_functions import categorical_cross_entropy, binary_cross_entropy

class TestCreateModel(unittest.TestCase):
    """Unit tests for create_model module"""
    def setUp(self):
        """Set up test environment"""
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--shape', type=int, nargs='+')
        self.parser.add_argument('--layer', type=int)
        self.parser.add_argument('--neurons', type=int)
        self.parser.add_argument('--conf', type=str)
        self.parser.add_argument('--features', type=str)
        self.parser.add_argument('--loss', type=str)
        self.parser.add_argument('--epoch', type=int)
        self.parser.add_argument('--learning_rate', type=float)
        self.parser.add_argument('--batch', type=int)
        self.parser.add_argument('--seed', type=int)
        self.parser.add_argument('--train_ratio', type=float)
        self.parser.add_argument('--outfile', type=str)
        self.parser.add_argument('dataset', type=str)

    def test_init_model_template(self):
        """Test initialization of model template"""
        expected = {
            'epoch': None,
            'alpha': None,
            'batch': None,
            'loss': None,
            'seed': None,
            'optimizer': None,
            'features': None,
            'target': None,
            'input': {
                'shape': None,
                },
            'layers': [],
            'output': {
                'shape': None,
                'activation': None,
                'weights_initializer': None
                }
            }
        self.assertDictEqual(init_model_template(), expected)

    def test_valid_fill_model_from_json(self):
        template = {
            'epoch': None,
            'alpha': None,
            'batch': None,
            'seed': None,
            'features': None,
            'target': None,
            'loss': None,
            'input': {
                'shape': None
                },
            'layers': [],
            'output': {
                'shape': None,
                'activation': None,
                'weights_initializer': None
                }
            }
        config_file_path = "tests/test_data/valid_model_config.json"
        expected = {
            "alpha": 0.123,
            "epoch": 123,
            "batch": 12,
            "seed": 32,
            "features": [
                "radius_mean",
                "texture_mean",
                "perimeter_mean",
                "area_mean",
                "smoothness_mean",
                "compactness_mean",
                "concavity_mean",
                "concave_points_mean",
                "symmetry_mean",
                "fractal_dimension_mean"
            ],
            "target": "diagnosis",
            "loss": None,
            "input": {
                "shape": 10
            },
            "output": {
                "activation": softmax,
                "weights_initializer": he_initialisation,
            },
            "layers": [
                {
                    "shape": 24,
                    "activation": sigmoid,
                    "weights_initializer": he_initialisation
                },
                {
                    "shape": 5,
                    "activation": sigmoid,
                    "weights_initializer": he_initialisation
                }
            ]
        }
        with open(config_file_path, 'r') as config_file:
            result = fill_model_from_json(template, config_file)
        self.assertDictEqual(result, expected)

    def test_invalid_fill_model_from_json_to_much_values(self):
        """Test filling model from invalid JSON config raises ValueError"""
        template = init_model_template()
        config_file_path = "tests/test_data/invalid_model_config_to_much_key.json"
        with self.assertRaises(KeyError):
            with open(config_file_path, 'r') as config_file:
                fill_model_from_json(template, config_file)

    def test_create_model_no_features(self):
        """Test that create_model auto-detects features when features=None"""
        args = argparse.Namespace(
            conf=None,
            epoch=100,
            alpha=0.001,
            train_ratio=0.8,
            batch=32,
            loss='categoricalCrossentropy',
            seed=42,
            optimizer='mini-batch',
            shape=[128, 64, 32],
            features=None,
            dataset='tests/test_data/data_training.csv'
        )
        target = 'diagnosis'
        result = create_model(args, target)
        
        # Test that features were auto-detected (all columns except target and id)
        expected_features = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
            'smoothness_mean', 'compactness_mean', 'concavity_mean', 
            'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_std', 'texture_std', 'perimeter_std', 'area_std', 
            'smoothness_std', 'compactness_std', 'concavity_std', 
            'concave_points_std', 'symmetry_std', 'fractal_dimension_std',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
            'smoothness_worst', 'compactness_worst', 'concavity_worst', 
            'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        self.assertEqual(result['features'], expected_features)
        self.assertEqual(result['input']['shape'], len(expected_features))
        self.assertIsNotNone(result['input']['train_data'])
        self.assertIsNotNone(result['input']['test_data'])
