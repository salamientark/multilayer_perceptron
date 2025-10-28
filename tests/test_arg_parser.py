"""
Unit tests for argument parsing in train.py

Tests both parse_args() and validate_args() functions to ensure
proper handling of command-line arguments and validation logic.
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
import argparse
from train import parse_args, validate_args


class TestArgParser(unittest.TestCase):
    """Test class for argument parsing functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Store original argv to restore after each test
        self.original_argv = sys.argv
    
    def tearDown(self):
        """Clean up after each test method"""
        # Always restore original argv
        sys.argv = self.original_argv
    
    def test_parse_args_with_valid_shape(self):
        """Test parsing valid arguments with --shape option"""
        # ARRANGE: Set up fake command line arguments
        sys.argv = [
            'train.py',
            'data_training.csv',
            '--shape', '10', '5',
            '--learning_rate', '0.009',
            '--seed', '43',
            '--epoch', '101',
            '--train_ratio', '0.75',
            '--batch', '16',
        ]
        
        # ACT: Call the function
        args = parse_args()
        
        # ASSERT: Check the results
        self.assertEqual(args.shape, [10, 5])
        self.assertEqual(args.dataset, 'data_training.csv')
        self.assertEqual(args.seed, 43)
        self.assertEqual(args.epoch, 101)
        self.assertEqual(args.alpha, 0.009)  # learning_rate stored as alpha
        self.assertEqual(args.batch, 16)
        self.assertEqual(args.train_ratio, 0.75)
    
    # Basic tests
    def test_parse_args_with_valid_layer_and_neurons(self):
        """Test parsing valid arguments with --layer and --neurons options"""
        sys.argv = [
            'train.py',
            '--learning_rate', '0.02',
            '--layer', '3',
            '--neurons', '8',
            '--seed', '44',
            '--epoch', '102',
            '--batch', '17',
            'data_training.csv'
        ]
        
        args = parse_args()
        
        self.assertEqual(args.layer, 3)
        self.assertEqual(args.neurons, 8)
        self.assertEqual(args.dataset, 'data_training.csv')
        self.assertEqual(args.seed, 44)
        self.assertEqual(args.epoch, 102)
        self.assertEqual(args.alpha, 0.02)
        self.assertEqual(args.batch, 17)

    def test_parse_args_with_conf_option(self):
        """Test parsing valid arguments with --conf option"""
        sys.argv = [
            'train.py',
            '--conf', 'model_config.json',
            'data_training.csv'
        ]
        
        args = parse_args()
        
        self.assertEqual(args.conf, 'model_config.json')
        self.assertEqual(args.dataset, 'data_training.csv')

    # TEST MISSING ARGS
    def test_parse_args_missing_dataset_raises_system_exit(self):
        """Test that missing dataset argument raises SystemExit"""
        sys.argv = [
            'train.py',
            '--shape', '10', '5',
            '--learning_rate', '0.009',
            '--seed', '43',
            '--epoch', '101',
            '--train_ratio', '0.75',
            '--batch', '16',
        ]
        
        # Should raise SystemExit when required positional arg is missing
        with self.assertRaises(SystemExit):
            parse_args()

    def test_validate_args_missing_seed_raises_exception(self):
        """Test that validate_args raises exception when seed is missing"""
        # Create a mock args object
        args = argparse.Namespace()
        args.conf = None
        args.seed = None  # Missing required parameter
        args.epoch = 100
        args.alpha = 0.01
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        args.dataset = "data_training.csv"
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Enter value for --seed/-s", str(context.exception))

    def test_validate_args_missing_epoch_raises_exception(self):
        """Test that validate_args raises exception when seed is missing"""
        # Create a mock args object
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = None  # Missing required parameter
        args.alpha = 0.01
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        args.dataset = "data_training.csv"
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Enter value for --epoch/-e", str(context.exception))

    def test_validate_args_missing_epoch_raises_exception(self):
        """Test that validate_args raises exception when seed is missing"""
        # Create a mock args object
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = None  # Missing required parameter
        args.alpha = 0.01
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        args.dataset = "data_training.csv"
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Enter value for --epoch/-e", str(context.exception))


    # TEST LEARNING RATE (alpha)
    def test_validate_args_learning_rate_to_high_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 100
        args.alpha = 1.5  # Invalid: must be in (0, 1)
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Learning rate must be in the range (0, 1]", str(context.exception))

    def test_validate_args_learning_rate_zero_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 100
        args.alpha = 0.0  # Invalid: must be in (0, 1)
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Learning rate must be in the range (0, 1]", str(context.exception))

    def test_validate_args_learning_rate_to_low_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 100
        args.alpha = -0.0  # Invalid: must be in (0, 1)
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Learning rate must be in the range (0, 1]", str(context.exception))

    # def test_validate_args_layer_without_neurons_raises_exception(self):
    #     """Test that validate_args raises exception when --layer used without --neurons"""
    #     args = argparse.Namespace()
    #     args.conf = None
    #     args.seed = 42
    #     args.epoch = 100
    #     args.alpha = 0.01
    #     args.batch = 16
    #     args.layer = 3      # Has layer
    #     args.neurons = None # Missing neurons
    #     args.shape = None
    #     args.train_ratio = 0.8
    #     
    #     with self.assertRaises(Exception) as context:
    #         validate_args(args)
    #     
    #     self.assertIn("--layer and --neurons MUST be used together", str(context.exception))


if __name__ == '__main__':
    # Run the tests when this file is executed directly
    unittest.main(verbosity=2)
