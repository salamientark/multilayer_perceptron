import numpy as np
import pandas as pd
from .ft_math import ft_min
from .network_layers import score_function


def batch_gradient_descent(thetas: np.ndarray,
                           features: pd.DataFrame,
                           target: np.ndarray,
                           alpha: float,
                           hypothesis=score_function
                           ) -> np.ndarray:
    """Calculate new thetas values using batch gradient descent

    Parameters:
      thetas (np.ndarray): Current thetas values
      features (pd.DataFrame): Features values
      target (np.ndarray): Target values (0 or 1)
      alpha (float): Learning rate
      hypothesis (function) (optionnal): Hypothesis function to use if not
                                         provided score_function will be used

    Returns:
      float: New theta value
    """
    new_thetas = thetas.copy()
    iction = np.array([hypothesis(thetas, row)
                       for row in features.to_numpy()])
    errors = iction - target
    gradient = np.dot(errors, features.to_numpy()) / len(features)
    new_thetas -= alpha * gradient
    return new_thetas


def stochastic_gradient_descent(thetas: np.ndarray,
                                features: pd.DataFrame,
                                target: np.ndarray,
                                alpha: float,
                                hypothesis=score_function
                                ) -> np.ndarray:
    """Calculate new thetas values using stochastic gradient descent

    Parameters:
      thetas (np.ndarray): Current thetas values
      features (pd.DataFrame): Features values
      target (np.ndarray): Target values (0 or 1)
      alpha (float): Learning rate
      hypothesis (function) (optionnal): Hypothesis function to use if not
                                         provided score_function will be used

    Returns:
      array: New theta value
    """

    # Shuffle dataset
    merged_df = features.copy()
    merged_df['target'] = target
    shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
    target = shuffled_df['target'].to_numpy()
    data = shuffled_df.drop(columns=['target']).to_numpy()
    new_thetas = thetas.copy()
    for data_row, target_row in zip(data, target):
        iction = hypothesis(new_thetas, data_row)  # Vector
        errors = iction - target_row  # Vector
        new_thetas -= alpha * errors * data_row  # Vector
    return new_thetas


def mini_batch_gradient_descent(thetas: np.ndarray,
                                features: pd.DataFrame,
                                target: np.ndarray,
                                alpha: float,
                                batch_size: int = 32,
                                hypothesis=score_function
                                ) -> np.ndarray:
    """Calculate new thetas values using mini batch gradient descent

    Parameters:
      thetas (np.ndarray): Current thetas values
      features (pd.DataFrame): Features values
      target (np.ndarray): Target values (0 or 1)
      alpha (float): Learning rate
      batch_size (int): Size of the mini batch
      hypothesis (function) (optionnal): Hypothesis function to use if not
                                         provided score_function will be used

    Returns:
      array: New theta value
    """
    # Shuffle dataset
    merged_df = features.copy()
    merged_df['target'] = target
    shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
    target = shuffled_df['target'].to_numpy()
    data = shuffled_df.drop(columns=['target']).to_numpy()
    new_thetas = thetas.copy()
    for start in range(0, len(data), batch_size):
        end = ft_min([start + batch_size, len(data)])
        data_batch = data[start:end]  # Matrix
        target_batch = target[start:end]  # Vector
        ictions = np.array([hypothesis(thetas, row) for row in data_batch])
        errors = ictions - target_batch
        gradient = np.dot(errors, data_batch) / len(data_batch)
        new_thetas -= alpha * gradient
    return new_thetas
