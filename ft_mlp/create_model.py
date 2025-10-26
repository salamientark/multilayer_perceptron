import json as json
import pandas as pd
from .network_layers import sigmoid, sigmoid_derivative, softmax
from .loss_functions import categorical_cross_entropy
from .model_utils import he_initialisation
from .preprocessing import split_dataset, standardize_df


FUNCTION_MAP = {
        'sigmoid': sigmoid,
        'softmax': softmax,
        'categoricalCrossentropy': categorical_cross_entropy,
        'heUniform': he_initialisation
        }


DERIVATIVE_MAP = {
        sigmoid: sigmoid_derivative
        }


def init_model_template() -> dict:
    """Initialize empty model template with default structure

    Creates a dictionary template for the tilayer perceptron model
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
            'seed': None,
            'data_train': None,
            'data_test': None,
            'optimizer': None,
            'input': {
                # 'features': [],
                'shape': None,
                },        # Input layer configuration
            'layers': [],         # List of hidden layer configurations
            'output': {
                'shape': None,
                'activation': None,
                'weights_initializer': None
                }          # Output layer configuration
            }
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
      dict: Dictionary reenting the layer with keys for shape,
            activation function, and weight initialization function
    """
    layer = {
            'shape': shape,
            'activation': (activation if activation is not None
                           else sigmoid),
            'weight_initializer': (weight_init if weight_init is not None
                                   else he_initialisation),
            'derivative': (DERIVATIVE_MAP[activation] if activation
                           is not None and activation in DERIVATIVE_MAP
                           else sigmoid_derivative)
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
    simple_keys = ['epoch', 'alpha', 'batch', 'loss', 'seed', 'inputs',
                   'optimizer']
    input_keys = model['input'].keys()
    layer_keys = model['output'].keys()
    function_keys = ['activation', 'weights_initializer', 'loss']
    for k, _ in conf.items():
        if k in conf and conf[k] is not None:
            if k in simple_keys:
                model[k] = (conf[k] if k not in function_keys
                            else FUNCTION_MAP[conf[k]])
            elif k == 'input':
                model[k] = {sub_k: (FUNCTION_MAP[val] if sub_k in function_keys
                                    else val)
                            for sub_k, val in conf[k].items()
                            if sub_k in input_keys}
            elif k == 'output':
                model[k] = {sub_k: (FUNCTION_MAP[val] if sub_k in function_keys
                                    else val)
                            for sub_k, val in conf[k].items() if sub_k
                            in layer_keys}
            else:  # output or layers
                model[k] = [{sub_k: (FUNCTION_MAP[val] if sub_k
                                     in function_keys else val)
                            for sub_k, val in layer.items() if sub_k
                             in layer_keys}
                            for layer in conf[k]]
    return model


def fill_model_from_param(args, model: dict) -> dict:
    """Initialize model parameters from CLI params

    Parameters
      args (argparse.Namespace): Parsed program arguments
      train_set (pandas.DataFrame): The training set

    Returns:
      dict: Model parameters
    """
    # Basic features overide
    args_dict = vars(args)
    for key, _ in model.items():
        if key == 'loss':
            model[key] = (FUNCTION_MAP[args_dict[key]] if args_dict[key]
                          is not None else model[key])
            continue
        model[key] = (args_dict[key] if key in args_dict
                      and args_dict[key] is not None
                      else model[key])
    # Fill model from args.shape
    if args.shape is not None:
        model['layers'] = [{
            'shape': n,
            'activation': sigmoid,
            'weights_initializer': he_initialisation
            } for n in args.shape]
    # Fill model from args.features
    if args.features is not None:
        model['input']['features'] = args.features
        model['input']['shape'] = len(model['input']['features'])
    return model


def fill_model_datasets(
        model: dict,
        dataset,
        training_rate: float,
        seed: int,
        target: str,
        features: list = []
        ) -> dict:
    """Split dataset and fill model with training and validation set

    Parameters:
      model (dict): Model parameters to fill with datasets
      dataset (pandas.DataFrame): The complete dataset
      training_rate (float): Ratio of the dataset to use for training
      seed (int): Seed for random operations to ensure reproducibility
      features (list, optional): List of feature column names to use.
                                 If empty, uses model['input']['features'].
      target (str): Name of the target column in the dataset``

    Returns:
      dict: Model parameters populated with training and validation datasets
    """
    df = pd.read_csv(dataset)
    if not features:
        features = model['input']['features']
    filtered_df = df[features + [target]]
    standardized_data = standardize_df(filtered_df)
    model['data_train'], model['data_test'] = split_dataset(
            standardized_data, ratio=training_rate, seed=seed)
    model['input']['train_data'] = model['data_train'][features].to_numpy()
    model['input']['test_data'] = model['data_test'][features].to_numpy()
    model['output']['activation'] = softmax
    model['output']['shape'] = filtered_df[target].nunique()
    return model


def create_model(args, target: str, features: list = []):
    """Create and initialize model parameters

    Parameters:
      args (argparse.Namespace): Parsed program arguments
    """
    model = init_model_template()
    if args.conf is not None:
        model = fill_model_from_json(model, args.conf)
    model = fill_model_from_param(args, model)
    model = fill_model_datasets(model, args.dataset, args.train_ratio,
                                model['seed'], target, features)
    # Set default loss function if not specified
    if model['loss'] is None:
        model['loss'] = FUNCTION_MAP['categoricalCrossentropy']
    model['optimizer'] = ('mini-batch' if model['batch'] is not None
                          else 'stochastic')
    # Set derivatives for each layer
    for layer in model['layers']:
        layer['derivative'] = DERIVATIVE_MAP[layer['activation']]
    return model


def load_model_from_json(file) -> dict:
    """Load model parameters from a JSON file

    Parameters:
      file : File object containing JSON model data

    Returns:
      dict: Model parameters loaded from JSON file
    """
    model = init_model_template()
    filled_model = fill_model_from_json(model, file)
    return filled_model
