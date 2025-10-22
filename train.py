import argparse as ap
import numpy as np
import pandas as pd
import json as json
import ft_mlp as ft_mlp

from types import FunctionType


class FunctionEncoder(json.JSONEncoder):
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
        return json.JSONEncoder.default(self, o)


# COLORS
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


# DEFAULT VALUES
FEATURES = [
        'radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave_points_mean', 'symmetry_mean',
        'fractal_dimension_mean', 'radius_std', 'texture_std',
        'perimeter_std', 'area_std', 'smoothness_std', 'compactness_std',
        'concavity_std', 'concave_points_std', 'symmetry_std',
        'fractal_dimension_std', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst',
        'symmetry_worst', 'fractal_dimension_worst'
        ]
TARGET = 'diagnosis'


def print_weights(model: dict):
    for i, layer in enumerate(model['layers']):
        print(f"Layer {i + 1} weights:{layer['weights']}")
        # print(f"Layer {i + 1} type:{type(layer['weights'])}")


def parse_args():
    """Parse program argument"""
    # Init parser
    parser = ap.ArgumentParser(prog="train.py",
                               description="Train a multilayer perceptron "
                                           "model.",
                               epilog=".... . .-.. .-.. --- / .-- --- .-. "
                                      ".-.. -.. -.-.--")

    # Create exclusion group for shape and layer
    shape_group = parser.add_mutually_exclusive_group(required=True)
    # Add parser argument
    shape_group.add_argument("--shape", type=int, nargs='+',
                             help="Define the number of neurons for each "
                                  "hidden layer.")
    shape_group.add_argument("--layer", type=int,
                             help="Define the number of hidden layers (use "
                                  "with --neurons).")
    shape_group.add_argument("--conf", type=ap.FileType('r'),
                             help="Model configuration file.")
    parser.add_argument("--features", choices=FEATURES, nargs='+',
                        default=FEATURES, help="List of features to use.")
    parser.add_argument("--neurons", type=int, required=False,
                        help="Define a constant number of neurons for all "
                             "hidden layers (use with --layer).")
    parser.add_argument("--loss", choices=['categoricalCrossentropy'],
                        # default="categoricalCrossentropy",
                        help="Loss function to use")
    parser.add_argument("--epoch", "-e", type=int,
                        # default=84,
                        help="Number of iteration of the trainig.")
    parser.add_argument("--learning_rate", "-a", type=float, dest="alpha",
                        # default=0.1,
                        help="Learning rate of the algorithm.")
    parser.add_argument("--batch", "-b", type=int, required=False,
                        # default=8,
                        help="Batch size if mini-batch gradient descent is "
                             "used.")
    parser.add_argument("--seed", "-s", type=int,
                        help="Seed to make model reproducible")
    parser.add_argument("--train_ratio", "-tr", type=float,
                        # default=0.8,
                        help="The part of the dataset used as the training "
                             "set. (validation set ratio = 1 - train_ratio)")
    parser.add_argument("--outfile", "-of", type=ap.FileType('w'),
                        default="weights.csv", help="Weight result file.")
    parser.add_argument("dataset", type=ap.FileType('r'),
                        help="Training dataset.")
    # Get args
    args = parser.parse_args()
    return args


def validate_args(args):
    """Check args to check validity

    Parameters:
      args: program parameters
    """
    if args.conf is None and args.seed is None:
        raise Exception("Enter value for --seed/-s.")
    if args.conf is None and args.epoch is None:
        raise Exception("Enter value for --epoch/-e.")
    if args.conf is None and args.train_ratio is None:
        raise Exception("Enter value for --train_ratio/-tr.")
    if args.conf is None and args.alpha is None:
        raise Exception("Enter value for --learning_rate/-a.")
    if args.conf is None and args.batch is None:
        raise Exception("Enter value for --batch/-b.")
    if (args.layer is None) != (args.neurons is None):
        raise Exception("--layer and --neurons MUST be used together")
    if args.shape is None and args.layer is None and args.conf is None:
        raise Exception("Either --shape or --layer and --neurons must be used")
    if args.shape is None and args.layer is not None:
        args.shape = [args.neurons] * args.layer
    if args.shape is not None and any(n <= 0 for n in args.shape):
        raise Exception("All layer must have at least one neuron.")
    if args.epoch is not None and args.epoch <= 0:
        raise Exception("Number of epoch must be > 0.")
    if args.alpha is not None and not (0 < args.alpha < 1):
        raise Exception("Learning rate must be in the range (0, 1]")
    if args.seed is not None and args.seed <= 0:
        raise Exception("Seed must be a positive integer")
    if args.train_ratio is not None and not (0 < args.train_ratio < 1):
        raise Exception("Train ratio must be between 0 and 1 excluded.")


def check_model(model: dict):
    """Check if the model is valid

    Will raise an Exception if the model is not valid.

    Parameters:
      model (dict): Model parameters to validate
    """
    if model['epoch'] is None or model['epoch'] <= 0:
        raise Exception("Number of epoch must be a positive integer.")
    if model['alpha'] is None or not (0 < model['alpha'] <= 1):
        raise Exception("Learning rate must be in the range (0, 1].")
    if model['batch'] is not None and model['batch'] <= 0:
        raise Exception("Batch size must be a positive integer.")
    if (model['loss'] is None or model['loss']
            is not ft_mlp.categorical_cross_entropy):
        raise Exception("Loss function must be categoricalCrossentropy.")
    if (model['seed'] is None or model['seed'] <= 0):
        raise Exception("Seed must be a positive integer.")
    if model['data_train'] is None or model['data_test'] is None:
        raise Exception("Training and validation datasets must be provided.")
    if model['input']['shape'] is None or model['input']['shape'] <= 0:
        raise Exception("Input layer must have a positive number of neurons.")
    if (model['input']['features'] is None
            or len(model['input']['features']) == 0):
        raise Exception("Input layer must have at least one feature.")
    if model['output']['shape'] is None or model['output']['shape'] <= 0:
        raise Exception("Output layer must have a positive number of neurons.")
    if (model['output']['activation'] is None or model['output']['activation']
            is not ft_mlp.softmax):
        raise Exception("Output layer activation must be softmax.")
    if (model['output']['weight_init'] is None or
            model['output']['weight_init'] is not ft_mlp.he_initialisation):
        raise Exception("Output layer weight initialization must be zero "
                        "initialization.")
    for layer in model['layers']:
        if layer['shape'] is None or layer['shape'] <= 0:
            raise Exception("All hidden layers must have a positive number of"
                            " neurons.")
        if (layer['activation'] is None
                or layer['activation'] is not ft_mlp.sigmoid):
            raise Exception("All hidden layers must use the sigmoid "
                            "activation function.")
        if (layer['weight_init'] is None or layer['weight_init']
                is not ft_mlp.he_initialisation):
            raise Exception("All hidden layers must use zero weight "
                            "initialization.")


def calculte_accuracy(predictions: np.ndarray, truth: np.ndarray) -> float:
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


def init_model(model: dict) -> dict:
    """Initialize model weights and bias

    Parameters:
      model (dict): Model parameters to initialize

    Returns:
      dict: Model with initialized weights and bias
    """
    seed = model['seed']
    data_inputs = len(model['input']['train_data'])
    for i, layer in enumerate(model['layers']):
        layer['gradients'] = {}
        if i == 0:
            layer['weights'], layer['bias'] = layer['weight_init'](
                    model['input']['shape'], layer['shape'], seed, data_inputs)
            continue
        layer['weights'], layer['bias'] = layer['weight_init'](
                model['layers'][i - 1]['shape'], layer['shape'], seed,
                data_inputs)
    model['output']['weights'], model['output']['bias'] = \
        model['output']['weight_init'](
                model['layers'][-1]['shape'], model['output']['shape'],
                seed, data_inputs
            )
    model['output']['gradients'] = {}
    model['output']['train_result'] = None
    model['output']['test_result'] = None
    model['output']['train_truth'] = ft_mlp.one_encoding(model['data_train'],
                                                         TARGET)
    model['output']['test_truth'] = ft_mlp.one_encoding(model['data_test'],
                                                        TARGET)
    model['output']['train_loss'] = []
    model['output']['test_loss'] = []
    model['output']['train_acc'] = []
    model['output']['test_acc'] = []
    return model


def feed_forward(model: dict):
    """Perform feed forwrd pass in the mlp

    Used in training calculate result for each layer

    Parameters:
      model (dict): Model parameters to use for feed forward pass
    """
    train_inputs = model['input']['train_data']
    test_inputs = model['input']['test_data']
    for layer in model['layers']:
        layer['result'] = ft_mlp.hidden_layer(
                train_inputs, layer['weights'],
                layer['bias'],
                activation=layer['activation'])
        train_inputs = layer['result']
        test_result = ft_mlp.hidden_layer(
                test_inputs, layer['weights'],
                layer['bias'],
                activation=layer['activation'])
        test_inputs = test_result
    train_predictions = ft_mlp.hidden_layer(
                train_inputs, model['output']['weights'],
                model['output']['bias'],
                activation=model['output']['activation'])
    test_predictions = ft_mlp.hidden_layer(
                test_inputs, model['output']['weights'],
                model['output']['bias'],
                activation=model['output']['activation'])

    # Update model
    train_truth = model['output']['train_truth']
    test_truth = model['output']['test_truth']
    model['output']['result'] = train_predictions
    model['output']['train_loss'].append(model['loss'](
            train_predictions,
            model['output']['train_truth']))
    model['output']['test_loss'].append(model['loss'](
            test_predictions,
            model['output']['test_truth']))
    model['output']['test_acc'].append(
            calculte_accuracy(train_predictions, train_truth))
    model['output']['train_acc'].append(
            calculte_accuracy(test_predictions, test_truth))


def backpropagation(model: dict):
    """Perform backpropagation to compute gradients

    Perform backpropagation to compute gradients for each layer in training
    process.

    Parameters:
      model (dict): Model parameters to use for backpropagation
    """
    predictions = model['output']['result']
    truth = model['output']['train_truth']
    gradient = predictions - truth  # Partial derivative (Crossentropy/softmax)
    gradient_weights_out = model['layers'][-1]['result'].T @ gradient
    gradient_bias_out = np.sum(gradient, axis=0)
    model['output']['gradients']['weights'] = gradient_weights_out
    model['output']['gradients']['bias'] = gradient_bias_out
    weights = model['output']['weights']
    for i in range(len(model['layers']) - 1, -1, -1):
        gradient = gradient @ weights.T
        gradient *= model['layers'][i]['derivative'](
                model['layers'][i]['result'])
        if i == 0:
            model['layers'][i]['gradients']['weights'] = \
                model['input']['train_data'].T @ gradient
        else:
            model['layers'][i]['gradients']['weights'] = \
                    model['layers'][i - 1]['result'].T @ gradient
        model['layers'][i]['gradients']['bias'] = np.sum(gradient, axis=0)
        weights = model['layers'][i]['weights']


def update_weights(model: dict):
    """Update model weights after training

    Parameters:
      model (dict): Model parameters to update
    """
    alpha = model['alpha']
    for layer in model['layers']:
        layer['weights'] -= alpha * layer['gradients']['weights']
        layer['bias'] -= alpha * layer['gradients']['bias']
    model['output']['weights'] -= alpha * \
        model['output']['gradients']['weights']
    model['output']['bias'] -= alpha * \
        model['output']['gradients']['bias']


def train(model: dict):
    """Perform the training of the model

    Parameters:
      model (dict): Model parameters to train
    """
    # Print param
    epoch_len = len(str(model['epoch']))
    print("data_train shape:", model['data_train'].shape)
    print("data_validation shape:", model['data_test'].shape)
    for i in range(model['epoch']):
        print(f"Epoch {i + 1:0{epoch_len}d}/{model['epoch']:0{epoch_len}d} - ",
              end="")
        feed_forward(model)
        backpropagation(model)
        update_weights(model)
        print(f"loss: {np.mean(model['output']['train_loss'][i]):.6f} - "
              f"val_loss: {np.mean(model['output']['test_loss'][i]):.6f}")


# def plot_loss_cuve(train_loss: np.ndarray, test_loss: np.ndarray):
#     """Plot loss curve for training and validation set
#
#     Parameters:
#       train_loss (np.ndarray): Training loss values
#       test_loss (np.ndarray): Validation loss values
#     """
    


def main(args):
    """Train the model"""
    model = ft_mlp.create_model(args, TARGET, FEATURES)
    check_model(model)  # Validate model inputs

    init_model(model)  # Init model weights and bias

    train(model)
    print("Good exit")
    return


if __name__ == "__main__":
    args = parse_args()
    try:
        validate_args(args)
        main(args)
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}")
