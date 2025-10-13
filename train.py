import argparse as ap
import numpy as np
import pandas as pd
import ft_datatools as ftdt
import json as json


# COLORS
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


# DEFAULT VALUES
FEATURES = [
        'radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean', 'symmetry_mean',
        'fractal_dimension_mean', 'radius_std', 'texture_std',
        'perimeter_std', 'area_std', 'smoothness_std', 'compactness_std',
        'concavity_std', 'concave points_std', 'symmetry_std',
        'fractal_dimension_std', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst'
        ]
TARGET = 'diagnosis'
FUNCTION_MAP = {
        'categoricalCrossentropy': lambda x : x,
        'sigmoid': ftdt.sigmoid,
        'softmax': ftdt.softmax
        }
MODEL_TEMPLATE = {
    "model": "multilayer perceptron",
    "alpha": 0.1,
    "epoch": 84,
    "batch": 8,
    "seed": 84,
    "features": None,
    "target": None,
    "output": {
        "activation": "softmax",
        "weight_init": "init_weights_zero"
    },
    "layers": [
        {
            "shape": 5,
            "activation": "sigmoid",
            "weight_init": "init_weights_zero"
        },
        {
            "shape": 5,
            "activation": "sigmoid",
            "weight_init": "init_weights_zero"
        }
    ]
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
    # print(args.layer is None)
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
    if args.conf is not None:
        return args


# model = {
#         'epoch': int,
#         'alpha': float,
#         'batch': int,
#         'loss': function,
#         'data_train': DataFrame
#         'data_test': DataFrame
#         'input': matrix,
#         'hidden': [[
#                     weights: matrix,
#                     acivation: matrix,
#                     weight_init: function
#                    ],
#                    [
#                      ...
#                    ]
#                   ]
#         'output': [
#                    weights: matrix,
#                    activation: function,
#                    weight_init: function
#                   ]
# }

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
            'input': None,        # Input layer configuration
            'layers': {},         # List of hidden layer configurations
            'output': {}          # Output layer configuration
            }
    return model


def fill_default_model_values(model: dict) -> dict:

    # model['output']['weight_init']
    return model


def fill_model_from_json(model: dict, config_file) -> dict:
    """Initalize model structure from provided json file

    Return:
      dict: Model Parameters
    """
    conf = json.load(config_file)
    print(json.dumps(conf, indent=4))
    print('\n\n\n')
    for key, _ in model.items():
        model[key] = conf[key] if key in conf else model[key]
    return model


def fill_model_from_param(args, input_size: int, output_size: int, model: dict) -> dict:
    """Initialize model parameters

    Parameters
      args (argparse.Namespace): Parsed program arguments
      input_size (int): Size of the input layer
      output_size (int): Size of the output layer
      train_set (pandas.DataFrame): The training set

    Returns:
      dict: Model parameters
    """
    # model = template

    # USER ADDED PARAM
    # Basic features overide
    # for key, _ in model.items():

    if args.batch is not None:
        model['batch'] = args.batch
    if args.epoch is not None:
        model['epoch'] = args.epoch
    if args.learning_rate is not None:
        model['alpha'] = args.learning_rate
    if args.loss is not None:
        model['loss'] = args.loss
    if args.seed is not None:
        model['seed'] = args.seed
    if args.features is not None:
        model['features'] = args.features

    # Input layer init

    # model['output'] = {
    #         'shape': output_size
    #         }

    # for i in range(len(shape))
    # model['layers'] = {
    #         'shape': 
    #         'weights': ftdt.init_weights_zero()

    # AUTOMATICALLY ADDED PARAM (=default)

    return model


def main(args):
    """Train the model"""
    print(args)
    model = init_model_template()
    if args.conf is not None:
        model = fill_model_from_json(model, args.conf)
    # model = init_model_from_param(args, 0, 0, model)
    print(json.dumps(model, indent=4))
    # model = init_model_from_param(args, )
    return

    #
    df = pd.read_csv(args.dataset)
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
