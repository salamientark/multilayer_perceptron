import numpy as np
from .ft_math import ft_max


def score_function(weights: np.ndarray,
                   features: np.ndarray,
                   bias: np.ndarray | float | None = None
                   ) -> np.ndarray | float:
    """Calculate score function for one sample
    thetas and features has to be the same size

    Parameters:
      weights (np.ndarray): Weights values
      features (np.ndarray): Features values
      bias (np.ndarray) (optionnal): Used if specified instead of feature
                                     first col

    Returns:
      np.ndarray | float: Score value array | Score value
    """
    res = weights @ features if bias is None else weights @ features + bias
    return res


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """Calculate sigmoid function for one sample

    Parameters:
      values (numpy.ndarray | float): Values to use for the sigmoid function

    Returns:
      np.ndarray | float: Sigmoid value
    """
    return np.where(values >= 0, 1 / (1 + np.exp(-values)),  # avoid overflow
                    np.exp(values) / (1 + np.exp(values)))


def sigmoid_derivative(sig_result: np.ndarray | float) -> np.ndarray | float:
    """Calculate sigmoid derivative function for matrix, array, sample

    BEWARE: input has to be the result of the sigmoid function
    This function is used in backpropagation to compute gradients

    Parameters:
      sig_result (numpy.ndarray | float): Values to use for the sigmoid
                                          derivative function

    Returns:
      numpy.ndarray | float: Sigmoid derivative value
    """
    return sig_result * (1 - sig_result)


def softmax(z: np.ndarray) -> np.ndarray:
    """Compute the softmax of a vector.
    Parameters:
      z (numpy.ndarray): Input vector.

    Return:
      numpy.ndarray: Softmax of the input vector.
    """
    z_max = ft_max(z)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1)[:, None]


def perceptron(inputs: np.ndarray,
               weights: np.ndarray,
               bias: float,
               activation=sigmoid
               ) -> np.ndarray:
    """Single perceptron

    Parameters:
      inputs (np.ndarray): Input data (matrix)
      weights (np.ndarray): Weights of the perceptron
      bias (float): Bias of the perceptron
      activation (function): Activation function to use

    Returns:
      np.ndarray: Output of the perceptron
    """
    weighted_sum = np.dot(weights, inputs) + bias
    return activation(weighted_sum)


def hidden_layer(inputs: np.ndarray,
                 weights: np.ndarray,
                 bias: np.ndarray,
                 activation=sigmoid
                 ) -> np.ndarray:
    """Compute the output of a hidden layer

    Parameters:
    input (np.ndarray): Input data (matrix)
    weights (np.ndarray): Weights of the layer (matrix)
    bias (np.ndarray): Bias of the layer
    activation (function) (optionnal): Activation function to use (default:
                                       sigmoid)

    Returns:
      np.ndarray: Output of the layer (matrix)
    """
    weighted_sum = inputs @ weights + bias
    return activation(weighted_sum)


def predict(model: dict, inputs: np.ndarray) -> np.ndarray:
    """Predict output for given inputs

    Used to just calculate the output of the model for given inputs.
    DO NOT USE as feed forward pass

    Parameters:
      model (dict): Model parameters to use for prediction
      inputs (np.ndarray): Input data to predict

    Returns:
      np.ndarray: Model predictions
    """
    layer_inputs = np.copy(inputs)
    result = None

    # Hidden layer calculation
    for layer in model['layers']:
        result = hidden_layer(
                layer_inputs, layer['weights'],
                layer['bias'],
                activation=layer['activation'])
        layer_inputs = result

    # Ouput layer calculation
    output_layer = model['output']
    result = hidden_layer(
            layer_inputs, output_layer['weights'],
            output_layer['bias'],
            activation=output_layer['activation'])
    return result
