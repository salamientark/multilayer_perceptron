"""
Unit tests for fill_model_optimizer function in ft_mlp/create_model.py

Tests cover:
1. If model optimizer is None, fill it based on batch size
2. Batch size logic and optimizer assignment:
   - batch == 1 -> optimizer = 'stochastic'
   - batch >= model['data_train'].shape[0] -> optimizer = 'batch_gradient_descent'
   - else -> optimizer = 'mini-batch'
3. If optimizer != None, it must be in ['adam', 'nesterov', 'RMSprop']
"""

import unittest
import numpy as np
import pandas as pd
from ft_mlp.create_model import fill_model_optimizer, init_model_template


class TestFillModelOptimizer(unittest.TestCase):
    """Unit tests for fill_model_optimizer function"""
    
    def setUp(self):
        """Set up test environment with sample model and data"""
        self.model_template = init_model_template()
        
        # Create sample training data for testing
        sample_data = np.random.rand(100, 5)  # 100 samples, 5 features
        self.sample_df = pd.DataFrame(sample_data)
        self.sample_df.columns = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        # Add a target column
        self.sample_df['target'] = np.random.choice(['A', 'B'], size=100)
        
        # Basic model with training data
        self.base_model = {
            **self.model_template,
            'data_train': self.sample_df,
            'optimizer': None,
            'batch': None
        }

    def test_optimizer_none_batch_none_default_to_batch_gradient_descent(self):
        """Test that when optimizer=None and batch=None, sets batch=data_size and optimizer='batch_gradient_descent'"""
        model = self.base_model.copy()
        model['optimizer'] = None
        model['batch'] = None
        
        fill_model_optimizer(model)
        
        expected_batch = model['data_train'].shape[0]  # Should be 100
        self.assertEqual(model['batch'], expected_batch)
        self.assertEqual(model['optimizer'], 'batch_gradient_descent')

    def test_optimizer_none_batch_1_sets_stochastic(self):
        """Test that when optimizer=None and batch=1, sets optimizer='stochastic'"""
        model = self.base_model.copy()
        model['optimizer'] = None
        model['batch'] = 1
        
        fill_model_optimizer(model)
        
        self.assertEqual(model['batch'], 1)
        self.assertEqual(model['optimizer'], 'stochastic')

    def test_optimizer_none_batch_greater_than_data_size_sets_batch_gradient_descent(self):
        """Test that when batch > data_train.shape[0], adjusts batch and sets optimizer='batch_gradient_descent'"""
        model = self.base_model.copy()
        model['optimizer'] = None
        model['batch'] = 150  # Greater than 100 (data size)
        
        fill_model_optimizer(model)
        
        expected_batch = model['data_train'].shape[0]  # Should be adjusted to 100
        self.assertEqual(model['batch'], expected_batch)
        self.assertEqual(model['optimizer'], 'batch_gradient_descent')

    def test_optimizer_none_batch_equal_to_data_size_sets_batch_gradient_descent(self):
        """Test that when batch == data_train.shape[0], sets optimizer='batch_gradient_descent'"""
        model = self.base_model.copy()
        model['optimizer'] = None
        model['batch'] = 100  # Equal to data size
        
        fill_model_optimizer(model)
        
        self.assertEqual(model['batch'], 100)
        self.assertEqual(model['optimizer'], 'batch_gradient_descent')

    def test_optimizer_none_batch_between_1_and_data_size_sets_mini_batch(self):
        """Test that when 1 < batch < data_train.shape[0], sets optimizer='mini-batch'"""
        model = self.base_model.copy()
        model['optimizer'] = None
        model['batch'] = 32  # Between 1 and 100
        
        fill_model_optimizer(model)
        
        self.assertEqual(model['batch'], 32)
        self.assertEqual(model['optimizer'], 'mini-batch')

    def test_optimizer_valid_adam_unchanged(self):
        """Test that valid optimizer 'adam' remains unchanged"""
        model = self.base_model.copy()
        model['optimizer'] = 'adam'
        model['batch'] = 32
        
        fill_model_optimizer(model)
        
        self.assertEqual(model['optimizer'], 'adam')
        self.assertEqual(model['batch'], 32)

    def test_optimizer_valid_nesterov_unchanged(self):
        """Test that valid optimizer 'nesterov' remains unchanged"""
        model = self.base_model.copy()
        model['optimizer'] = 'nesterov'
        model['batch'] = 64
        
        fill_model_optimizer(model)
        
        self.assertEqual(model['optimizer'], 'nesterov')
        self.assertEqual(model['batch'], 64)

    def test_optimizer_valid_rmsprop_unchanged(self):
        """Test that valid optimizer 'RMSprop' remains unchanged"""
        model = self.base_model.copy()
        model['optimizer'] = 'RMSprop'
        model['batch'] = 16
        
        fill_model_optimizer(model)
        
        self.assertEqual(model['optimizer'], 'RMSprop')
        self.assertEqual(model['batch'], 16)

    def test_optimizer_invalid_raises_error(self):
        """Test that invalid optimizer raises ValueError"""
        model = self.base_model.copy()
        model['optimizer'] = 'invalid_optimizer'
        model['batch'] = 32
        
        with self.assertRaises(ValueError) as context:
            fill_model_optimizer(model)
        
        self.assertIn("Optimizer must be one of ['adam', 'nesterov', 'RMSprop']", str(context.exception))

    def test_optimizer_sgd_invalid_raises_error(self):
        """Test that 'sgd' optimizer (not in allowed list) raises ValueError"""
        model = self.base_model.copy()
        model['optimizer'] = 'sgd'
        model['batch'] = 32
        
        with self.assertRaises(ValueError) as context:
            fill_model_optimizer(model)
        
        self.assertIn("Optimizer must be one of ['adam', 'nesterov', 'RMSprop']", str(context.exception))

    def test_batch_adjustment_when_greater_than_data_size(self):
        """Test batch size adjustment when it exceeds training data size"""
        model = self.base_model.copy()
        model['optimizer'] = None
        model['batch'] = 200  # Much greater than 100
        
        fill_model_optimizer(model)
        
        # Batch should be adjusted to match data size
        self.assertEqual(model['batch'], model['data_train'].shape[0])
        self.assertEqual(model['optimizer'], 'batch_gradient_descent')

    def test_edge_case_batch_zero_or_negative(self):
        """Test edge case where batch is zero or negative"""
        # Test batch = 0
        model = self.base_model.copy()
        model['optimizer'] = None
        model['batch'] = 0
        
        with self.assertRaises(ValueError) as context:
            fill_model_optimizer(model)
        
        self.assertIn("Batch size must be positive", str(context.exception))
        
        # Test negative batch
        model['batch'] = -5
        with self.assertRaises(ValueError) as context:
            fill_model_optimizer(model)
        
        self.assertIn("Batch size must be positive", str(context.exception))

    def test_empty_training_data_raises_error(self):
        """Test that empty training data raises an appropriate error"""
        model = self.base_model.copy()
        model['data_train'] = pd.DataFrame()  # Empty DataFrame
        model['optimizer'] = None
        model['batch'] = 32
        
        with self.assertRaises(ValueError) as context:
            fill_model_optimizer(model)
        
        self.assertIn("Training data cannot be empty", str(context.exception))

    def test_missing_data_train_key_raises_error(self):
        """Test that missing 'data_train' key raises KeyError"""
        model = self.base_model.copy()
        del model['data_train']  # Remove data_train key
        model['optimizer'] = None
        model['batch'] = 32
        
        with self.assertRaises(KeyError) as context:
            fill_model_optimizer(model)
        
        self.assertIn("'data_train'", str(context.exception))


class TestFillModelOptimizerIntegration(unittest.TestCase):
    """Integration tests for fill_model_optimizer with real data structure"""
    
    def setUp(self):
        """Set up with realistic model structure"""
        # Create a more realistic model structure similar to what create_model produces
        self.realistic_model = {
            'epoch': 100,
            'alpha': 0.01,
            'batch': None,
            'loss': None,
            'seed': 42,
            'optimizer': None,
            'features': ['feature1', 'feature2', 'feature3'],
            'target': 'diagnosis',
            'input': {'shape': 3},
            'layers': [],
            'output': {'shape': 2, 'activation': None, 'weights_initializer': None},
            'data_train': pd.DataFrame({
                'feature1': np.random.rand(50),
                'feature2': np.random.rand(50), 
                'feature3': np.random.rand(50),
                'diagnosis': np.random.choice(['M', 'B'], 50)
            })
        }

    def test_integration_with_realistic_model_structure(self):
        """Test fill_model_optimizer with realistic model structure"""
        model = self.realistic_model.copy()
        
        # Test default case (None optimizer, None batch)
        fill_model_optimizer(model)
        
        self.assertEqual(model['batch'], 50)  # Should match data size
        self.assertEqual(model['optimizer'], 'batch_gradient_descent')
        
    def test_integration_mini_batch_scenario(self):
        """Test integration with mini-batch scenario"""
        model = self.realistic_model.copy()
        model['batch'] = 16  # Mini-batch size
        
        fill_model_optimizer(model)
        
        self.assertEqual(model['batch'], 16)
        self.assertEqual(model['optimizer'], 'mini-batch')


if __name__ == '__main__':
    unittest.main()