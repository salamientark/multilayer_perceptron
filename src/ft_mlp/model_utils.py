import numpy as np
import pandas as pd
from .ft_math import ft_argmax
from .network_layers import sigmoid, softmax
from .loss_functions import categorical_cross_entropy
from .colors import BLUE, GREEN, RESET
from .initializer import he_initialisation
from random import seed, randrange
from sys import maxsize
from json import dump, dumps, JSONEncoder
from types import FunctionType


class FunctionEncoder(JSONEncoder):
    """Custom JSON encoder to handle function objects."""
    def default(self, o):
        if callable(o):
            return o.__name__
        elif isinstance(o, FunctionType):
            return o.__name__
        elif isinstance(o, pd.DataFrame):
            return o.shape
        elif isinstance(o, np.ndarray):
            return o.shape
        elif isinstance(o, pd.Series):
            return o.shape
        return JSONEncoder.default(self, o)


def print_model(model: dict) -> None:
    """Print model summary to the console

    Parameters:
      model (dict): Model parameters
    """
    print(dumps(model, indent=4, cls=FunctionEncoder))


def get_random_seed() -> int:
    """Generate a random positive int as seed

    Returns:
      int: Generated seed
    """
    seed()
    return randrange(1, maxsize)


def get_random_batch_indexes(
        data_size: int,
        seed: int | None = None) -> np.ndarray:
    """Get random batch indexes for mini-batch gradient descent

    Parameters:
      data_size (int): Size of the dataset
    seed (int) (optional) : Seed for random generator
                            (train will use seed + actual epoch for
                            reproducible results)

    Returns:
      np.ndarray: Random batch indexes
    """
    rng = np.random.default_rng() if seed is None \
        else np.random.default_rng(seed)
    permutated_indexes = rng.permutation(data_size)
    return permutated_indexes


def save_weights(filename: str, model: dict):
    """Save model weights to a npz file

    npz file are numpy compressed files containing arrays.

    Parameters:
      filename (str): Output filename
      model (dict): Model parameters to save
    """
    weights = {}
    for i, layer in enumerate(model['layers']):
        weights[f'layer_{i}_weights'] = layer['weights']
        weights[f'layer_{i}_bias'] = layer['bias']
    weights['output_weights'] = model['output']['weights']
    weights['output_bias'] = model['output']['bias']
    print(f"Saving weights to {BLUE}{filename}{RESET}... ", end="")
    np.savez(filename, **weights)
    print(f"{GREEN}Success{RESET}")


def load_weights_from_file(file) -> dict:
    """Load model weights from file

    Used for prediction.

    Parameters:
      file (str | file type) : File path or opened file IO buffer

    Returns:
      dict : Weights loaded from file
    """
    weights = np.load(file)
    return weights


def save_model(filename: str, model: dict):
    """Save model to a json file

    Parameters:
      filename (str): Output filename
      model (dict): Model parameters to save
    """
    model_template = {}

    model_template['epoch'] = model['epoch']
    model_template['alpha'] = model['alpha']
    model_template['batch'] = (model['batch'] if model['batch'] <
                               len(model['data_train'])
                               else len(model['data_train']))
    model_template['loss'] = FUNCTION_NAME[model['loss']]
    model_template['seed'] = model['seed']
    model_template['optimizer'] = model['optimizer']
    model_template['features'] = model['features']
    model_template['target'] = model['target']
    # Output column i means classes[i]. Without this, predict would re-derive
    # the order from the prediction file and could invert the classes.
    model_template['classes'] = model['classes']
    # Fitted on the training set. Predict must reuse these rather than
    # recompute them from the prediction file.
    model_template['standardization'] = model['standardization']

    model_template['input'] = {}
    model_template['input']['shape'] = model['input']['shape']

    model_template['layers'] = []
    for layer in model['layers']:
        filtered_layer = {}
        filtered_layer['shape'] = layer['shape']
        filtered_layer['activation'] = FUNCTION_NAME[
                layer['activation']]
        filtered_layer['weights_initializer'] = FUNCTION_NAME[
                layer['weights_initializer']]
        model_template['layers'].append(filtered_layer.copy())

    model_template['output'] = {}
    model_template['output']['shape'] = model['output']['shape']
    model_template['output']['activation'] = FUNCTION_NAME[
            model['output']['activation']]
    model_template['output']['weights_initializer'] = \
        FUNCTION_NAME[model['output']['weights_initializer']]

    print(f"Saving model to {BLUE}{filename}{RESET}... ", end="")
    with open(filename, 'w') as f:
        dump(model_template, f, indent=4)
    print(f"{GREEN}Success{RESET}")


def calculate_accuracy(predictions: np.ndarray, truth: np.ndarray) -> float:
    """Calculate accuracy of predictions

    Parameters:
      predictions (np.ndarray): Model predictions
      truth (np.ndarray): Ground truth labels

    Returns:
      float: Accuracy value
    """
    prediction_indexes = ft_argmax(predictions)
    truth_indexes = ft_argmax(truth)
    good_prediction = (prediction_indexes == truth_indexes)
    accuracy = np.sum(good_prediction) / len(good_prediction)
    return accuracy


def calculate_loss(
        predictions: np.ndarray,
        truth: np.ndarray,
        loss) -> np.ndarray:
    """Calculate loss for given predictions and truth

    Calculate loss for each inputs so loss can be averaged later.

    Parameters:
      predictions (np.ndarray): Model predictions
      truth (np.ndarray): Ground truth labels
      loss (function): Loss function to use

    Returns:
      np.ndarray: Loss for each input
    """
    return loss(predictions, truth)


def calculate_loss_mean(
        predictions: np.ndarray,
        truth: np.ndarray,
        loss) -> float:
    """Calculate loss mean for given predictions and truth

    Calculate loss mean to show training progress.

    Parameters:
      predictions (np.ndarray): Model predictions
      truth (np.ndarray): Ground truth labels
      loss (function): Loss function to use

    Returns:
      float: Mean loss value
    """
    all_loss = loss(predictions, truth)
    return np.sum(all_loss) / len(all_loss)


FUNCTION_NAME = {
        sigmoid: 'sigmoid',
        softmax: 'softmax',
        categorical_cross_entropy: 'categoricalCrossentropy',
        he_initialisation: 'heUniform'
        }
