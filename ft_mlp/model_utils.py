import numpy as np
import pandas as pd
from .ft_math import ft_mean, ft_std
from random import seed, randrange
from sys import maxsize


def get_random_seed() -> int:
    """Generate a random positive int as seed

    Returns:
      int: Generated seed
    """
    seed()
    return randrange(1, maxsize)


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


def he_initialisation(features: int, output: int, seed: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Initialize a weight matrix with He initialization

    Parameters:
      feature (int): Number of features
      output (int): Number of outputs (next layer neurons nbr)
      seed (int): Random seed

    Returns:
        tuple(np.ndarray, np.ndarray): Weights matrix and bias
    """
    rng = np.random.default_rng(seed=seed)
    return (rng.standard_normal(size=(features, output)) *
            np.sqrt(2 / output), np.zeros(output))


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
    means = {feature: ft_mean(df[feature]) for feature in features}
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
