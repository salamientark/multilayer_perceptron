import numpy as np
import pandas as pd
from .ft_math import ft_mean, ft_std
from .network_layers import sigmoid, softmax
from .loss_functions import categorical_cross_entropy
from .colors import BLUE, GREEN, RESET
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
                            (train will use seed * actual epoch for
                            reproducible results)

    Returns:
      np.ndarray: Random batch indexes
    """
    rng = np.random.default_rng() if seed is None \
        else np.random.default_rng(seed)
    permutated_indexes = rng.permutation(data_size)
    return permutated_indexes


def init_thetas(classes: list, feature_nbr: int) -> dict:
    """Initialize thetas dictionary with zeros

    Parameters:
      classes (list): List of class names
      feature_nbr (int): Number of features

    Returns:
      (dict): theta parameters for each class
    """
    thetas = {cls: np.zeros(feature_nbr) for cls in classes}
    return thetas


def init_weights_zero(features: int, output: int) -> tuple[np.ndarray, float]:
    """Initialize a weight matrix with zeros

    Parameters:
      feature (int): Number of features
      output (int): Number of outputs (next layer neurons nbr)

    Returns:
        tuple(np.ndarray, float): Weights matrix and bias
    """
    return np.zeros((features, output)), 0.0


def he_initialisation(features: int, output: int, seed: int, inputs: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Initialize a weight matrix with He initialization

    Parameters:
      feature (int): Number of features
      output (int): Number of outputs (next layer neurons nbr)
      inputs (int): Number of data inputs
      seed (int): Random seed

    Returns:
        tuple(np.ndarray, np.ndarray): Weights matrix and bias
    """
    rng = np.random.default_rng(seed=seed)
    return (rng.normal(loc=0.0, scale=np.sqrt(2 / inputs),
                       size=(features, output)), np.zeros(output))


def unstandardized_thetas(
        thetas: dict,
        df: pd.DataFrame,
        features: list
        ) -> dict:
    """Convert standardized thetas to unstandardized thetas

    Parameters:
      thetas (dict): Standardized thetas
      df (pd.DataFrame): Dataframe with original data
      features (list): List of features names

    Returns:
      dict: Unstandardized thetas
    """
    means = {feature: ft_mean(df[feature].to_numpy()) for feature in features}
    std = {feature: ft_std(df[feature], mean=means[feature])
           for feature in features}
    unstandardized = {
        cls: [
            theta[0] - sum(
                (theta[i] * means[features[i - 1]]) / std[features[i - 1]]
                for i in range(1, len(theta))
            )
        ] + [
            theta[i] / std[features[i - 1]]
            for i in range(1, len(theta))
        ]
        for cls, theta in thetas.items()
    }
    return unstandardized


def save_thetas(thetas: dict, features: list) -> None:
    """Save thetas to a file

    Parameters:
      thetas (dict): Thetas to save
      features (list): List of features names
    """
    with open("thetas.csv", "w") as f:
        f.write("Class,Bias," + ",".join(features) + "\n")
        for cls, theta in thetas.items():
            f.write(cls + "," + ",".join([str(t) for t in theta]) + "\n")


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
    prediction_indexes = np.argmax(predictions, axis=1)
    truth_indexes = np.argmax(truth, axis=1)
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
