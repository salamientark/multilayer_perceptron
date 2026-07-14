import numpy as np


def he_initialisation(features: int, output: int, seed: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Initialize a weight matrix with He uniform initialization

    Weights are drawn uniformly from [-limit, limit] with
    limit = sqrt(6 / fan_in), where fan_in is the number of inputs of the
    layer (its number of features), not the number of samples in the dataset.

    Parameters:
      features (int): Number of features of the layer (fan_in)
      output (int): Number of outputs (next layer neurons nbr)
      seed (int): Random seed

    Returns:
        tuple(np.ndarray, np.ndarray): Weights matrix and bias
    """
    rng = np.random.default_rng(seed=seed)
    limit = np.sqrt(6 / features)
    return (rng.uniform(low=-limit, high=limit, size=(features, output)),
            np.zeros(output))
