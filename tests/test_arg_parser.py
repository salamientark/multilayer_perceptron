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
import io
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

    # TEST parse_args MISSING ARGS
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
        
        captured_stderr = io.StringIO()
        sys.stderr = captured_stderr
        
        with self.assertRaises(SystemExit) as context:
            parse_args()
        
        sys.stderr = sys.__stderr__  # Restore stderr
        
        # Check exit code (argparse exits with 2 for argument errors)
        self.assertEqual(context.exception.code, 2)
        
        # Check the error message
        error_message = captured_stderr.getvalue()
        self.assertIn("error: the following arguments are required: dataset", error_message)

    def test_parse_args_missing_shape_raises_exception(self):
        """Test that parse_args raises exception when shape/neurons is missing"""
        
        # Mock sys.argv to simulate command line arguments
        # Only provide arguments that will cause the error
        sys.argv = [
            'train.py',
            '--dataset', 'data_training.csv',
            '--epoch', '100',
            '--alpha', '0.01',
            '--batch', '16',
            '--train-ratio', '0.8'
            # Intentionally omit --shape, --layer, and --conf
        ]
        
        # Capture stderr to check error message
        captured_stderr = io.StringIO()
        sys.stderr = captured_stderr
        
        with self.assertRaises(SystemExit) as context:
            parse_args()
        
        sys.stderr = sys.__stderr__  # Restore stderr
        
        # Check exit code (argparse exits with 2 for argument errors)
        self.assertEqual(context.exception.code, 2)
        
        # Check the error message
        error_message = captured_stderr.getvalue()
        self.assertIn("one of the arguments --shape --layer --conf is required", error_message)

    # TEST validate_args MISSING ARGS
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
        """Test that validate_args raises exception when epoch is missing"""
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

    def test_validate_args_missing_alpha_raises_exception(self):
        """Test that validate_args raises exception when epoch is missing"""
        # Create a mock args object
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 100
        args.alpha = None
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        args.dataset = "data_training.csv"
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Enter value for --learning_rate/-a", str(context.exception))

    # TEST validate_args INVALID ARGS
    def test_validate_args_seed_zero_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 0
        args.epoch = 100
        args.alpha = 0.1
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Seed must be a positive integer", str(context.exception))

    def test_validate_args_seed_to_low_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = -1
        args.epoch = 100
        args.alpha = 0.1  # Invalid: must be in (0, 1)
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Seed must be a positive integer", str(context.exception))

    def test_validate_args_epoch_to_low_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = -19
        args.alpha = 0.1
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Number of epoch must be > 0.", str(context.exception))

    def test_validate_args_epoch_to_zero_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 0
        args.alpha = 0.1
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Number of epoch must be > 0.", str(context.exception))

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
        args.alpha = -0.1  # Invalid: must be in (0, 1)
        args.batch = 16
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Learning rate must be in the range (0, 1]", str(context.exception))

    def test_validate_args_batch_to_low_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 100
        args.alpha = 0.1
        args.batch = -32
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Batch size must be a positive integer.", str(context.exception))

    def test_validate_args_batch_zero_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 100
        args.alpha = 0.1
        args.batch = 0
        args.layer = None
        args.neurons = None
        args.shape = [10, 5]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("Batch size must be a positive integer.", str(context.exception))

    def test_validate_args_negative_shape_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 100
        args.alpha = 0.1
        args.batch = 0
        args.layer = None
        args.neurons = None
        args.shape = [-1, 5, 10]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("All layer must have at least one neuron.", str(context.exception))

    def test_validate_args_zero_shape_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 100
        args.alpha = 0.1
        args.batch = 0
        args.layer = None
        args.neurons = None
        args.shape = [10, 0, 10]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("All layer must have at least one neuron.", str(context.exception))

    def test_validate_args_neg_last_shape_raises_exception(self):
        """Test that validate_args raises exception for invalid learning rate"""
        args = argparse.Namespace()
        args.conf = None
        args.seed = 42
        args.epoch = 100
        args.alpha = 0.1
        args.batch = 0
        args.layer = None
        args.neurons = None
        args.shape = [10, 5, -1]
        args.train_ratio = 0.8
        
        with self.assertRaises(Exception) as context:
            validate_args(args)
        
        self.assertIn("All layer must have at least one neuron.", str(context.exception))

    # TEST conf FILE

if __name__ == '__main__':
    # Run the tests when this file is executed directly
    unittest.main(verbosity=2)
