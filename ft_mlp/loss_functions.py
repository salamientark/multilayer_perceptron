import numpy as np


def categorical_cross_entropy(prediction: np.ndarray, truth: np.ndarray):
    """Calculate CrossEntropy for model tunning (loss)

    Parameters:
      prediction (np.ndarray): Values predicted by the model
      truth (np.ndarray): True value from the dataset

    Return:
      (np.ndarray): CrossEntropy result for each input
    """
    epsilon = 1e-9
    clipped_prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
    loged_prediction = np.log(clipped_prediction)
    loss = truth * loged_prediction
    return -np.sum(loss, axis=1)


def binary_cross_entropy(prediction: np.ndarray, truth: np.ndarray):
    """Calculate BinaryCrossEntropy for model tunning (loss)

    Parameters:
      prediction (np.ndarray): Values predicted by the model
      truth (np.ndarray): True value from the dataset

    Return:
      (np.ndarray): CrossEntropy result for each input
    """
    epsilon = 1e-9
    clipped_prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
    logged_prediction = (truth * np.log(clipped_prediction) + (1. - truth) *
                         np.log(1. - clipped_prediction))
    return -np.sum(logged_prediction, axis=1) / len(prediction[0])
