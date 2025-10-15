import argparse as ap
import numpy as np
import pandas as pd
import ft_datatools as ftdt
import json as json

import split_dataset


class FunctionEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle function objects."""
    def default(self, o):
        if callable(o):
            return o.__name__
        elif isinstance(o, pd.DataFrame):
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
FUNCTION_MAP = {
        'sigmoid': ftdt.sigmoid,
        'softmax': ftdt.softmax,
        'categoricalCrossentropy': ftdt.categorical_cross_entropy
        }


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
                        # default=ftdt.get_random_seed(),
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


def init_model_template() -> dict:
    """Initialize empty model template with default structure

    Creates a dictionary template for the multilayer perceptron model
    with all required fields set to None or empty lists.

    Return:
        dict: Empty model template with keys for epoch, alpha (learning rate),
              batch size, loss function, input layer, hidden layers, and
              output layer
    """
    model = {
            'epoch': None,        # Number of training iterations
            'alpha': None,        # Learning rate for gradient descent
            'batch': None,        # Batch size for mini-batch gradient descent
            'loss': None,         # Loss function to optimize
            'data_train': None,
            'data_test': None,
            'input': {
                'features': [],
                'shape': None
                },        # Input layer configuration
            'layers': [],         # List of hidden layer configurations
            'output': {
                'shape': None,
                'activation': None,
                'weight_init': None
                }          # Output layer configuration
            }
    return model


def fill_default_model_values(model: dict) -> dict:

    # model['output']['weight_init']
    return model


def create_model_layer(shape: int, activation=None, weight_init=None) -> dict:
    """Create a model layer for the multilayer perceptron

    Parameters:
      shape (int): Number of neurons in the layer
      activation (function, optional): Activation function for the layer.
                                 Defaults to sigmoid if None.
      weight_init (function, optional): Weight initialization function.
                                    Defaults to zero initialization if None.

    Returns:
      dict: Dictionary representing the layer with keys for shape,
            activation function, and weight initialization function
    """
    layer = {
            'shape': shape,
            'activation': activation if activation is not None
                                     else ftdt.sigmoid,
            'weight_init':weight_init if weight_init is not None
                                      else ftdt.init_weights_zero
            }
    return layer


def fill_model_from_json(model: dict, config_file) -> dict:
    """Initalize model structure from provided json file

    Parameters:
      model (dict): Base model template to fill with configuration values
      config_file: File object containing JSON configuration data

    Return:
      dict: Model Parameters populated from JSON configuration
    """
    conf = json.load(config_file)
    for k, _ in model.items():
        if k in conf and conf[k] is not None:
            print(k)
            if k not in ['input', 'output', 'layers']:
                model[k] = conf[k]
            elif k in ['input', 'output']:
                model[k] = {sub_k: val for sub_k, val in conf[k].items()}
            else:
                model[k] = [{sub_k: val for sub_k, val in layer.items()} for layer in conf[k]]
    return model


def fill_model_from_param(args, model: dict) -> dict:
    """Initialize model parameters

    Parameters
      args (argparse.Namespace): Parsed program arguments
      train_set (pandas.DataFrame): The training set

    Returns:
      dict: Model parameters
    """
    # Basic features overide
    args_dict = vars(args)
    for key, _ in model.items():
        model[key] = (args_dict[key] if key in args_dict 
                                        and args_dict[key] is not None 
                                     else model[key])
    # Fill model from args.shape
    if args.shape is not None:
        model['layers'] = [{
            'shape': n,
            'activation': ftdt.sigmoid,
            'weight_init': ftdt.init_weights_zero
            } for n in args.shape]
    # Fill model from args.features
    if args.features is not None:
        model['input']['features'] = args.features
        model['input']['shape'] = len(model['input']['features'])
    return model


def fill_model_datasets(model: dict, dataset, training_rate: float, seed: int) -> dict:
    """Split dataset and fill model with training and validation set

    Parameters:
      model (dict): Model parameters to fill with datasets
      dataset (pandas.DataFrame): The complete dataset
      training_rate (float): Ratio of the dataset to use for training
      seed (int): Seed for random operations to ensure reproducibility

    Returns:
      dict: Model parameters populated with training and validation datasets
    """
    # Extract dataset
    df = pd.read_csv(dataset)  # Read dataset file
    filtered_df = df[FEATURES + [TARGET]]  # Keep only interseting columns
    model['data_train'], model['data_test'] = ftdt.split_dataset(
            filtered_df, ratio=training_rate, seed=args.seed)
    # Fill output shape
    model['output']['activation'] = ftdt.softmax
    model['output']['weight_init'] = ftdt.init_weights_zero
    model['output']['shape'] = filtered_df[TARGET].nunique()
    return model


def check_model(model: dict):
    """Check if the model is valid"""
    

def main(args):
    """Train the model"""
    # print(args)
    model = init_model_template()
    if args.conf is not None:
        model = fill_model_from_json(model, args.conf)
    # print(json.dumps(model, indent=4), '\n\n\n')
    model = fill_model_from_param(args, model)
    # print(json.dumps(model, indent=4, cls=FunctionEncoder))

    # Extract dataset
    model = fill_model_datasets(model, args.dataset, args.train_ratio, args.seed)

    if model['loss'] is None:
        model['loss'] = FUNCTION_MAP['categoricalCrossentropy']
    # print(train_set, '\n\n\n')
    # print(validation_set)
    print(json.dumps(model, indent=4, cls=FunctionEncoder))


    loss = ftdt.categorical_cross_entropy(np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]),
                              np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    return
    #
    standardized = ftdt.standardize_df(df.drop(['id'],
                                               axis=1))
    standardized['diagnosis'] = ftdt.one_encoding(df, 'diagnosis')

    # Split dataset into train and test
    train_set, validation_set = ftdt.split_dataset(df,
                                                   ratio=args.train_ratio,
                                                   seed=args.seed)

    # Make model


    print(standardized)


if __name__ == "__main__":
    args = parse_args()
    try:
        validate_args(args)
        main(args)
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}")
