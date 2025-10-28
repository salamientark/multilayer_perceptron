import numpy as np


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
