import json as json
import pandas as pd
from .network_layers import sigmoid, sigmoid_derivative, softmax
from .loss_functions import categorical_cross_entropy
from .initializer import he_initialisation
from .preprocessing import (split_dataset, standardize_df, get_class_list,
                            get_standardization_stats)


FUNCTION_MAP = {
        'sigmoid': sigmoid,
        'softmax': softmax,
        'categoricalCrossentropy': categorical_cross_entropy,
        'heUniform': he_initialisation
        }


DERIVATIVE_MAP = {
        sigmoid: sigmoid_derivative
        }


def get_function(name: str):
    """Resolve a function name from a configuration file

    Parameters:
      name (str): Name of the function as written in the config file

    Returns:
      The matching function

    Raises:
      Exception: If the name is not a function the library implements
    """
    if name not in FUNCTION_MAP:
        known = ', '.join(sorted(FUNCTION_MAP))
        raise Exception(f"Unknown function '{name}' in the configuration "
                        f"file. Known functions: {known}.")
    return FUNCTION_MAP[name]


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
            'optimizer': None,
            'features': None,
            'target': None,
            'classes': None,      # Class order of the output layer columns
            'standardization': None,  # Per-feature mean/std fitted on train
            'input': {
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
                   'features', 'target', 'classes', 'standardization']
    input_keys = model['input'].keys()
    layer_keys = model['output'].keys()
    function_keys = ['activation', 'weights_initializer', 'loss']
    for k, _ in conf.items():
        if k == 'model' or k == 'optimizer':
            continue
        if k in model:
            if conf[k] is None:
                continue
            if k in simple_keys:
                model[k] = (conf[k] if k not in function_keys
                            else get_function(conf[k]))
            elif k == 'input':
                model[k] = {sub_k: (get_function(val) if sub_k in function_keys
                                    else val)
                            for sub_k, val in conf[k].items()
                            if sub_k in input_keys}
            elif k == 'output':
                model[k] = {sub_k: (get_function(val) if sub_k in function_keys
                                    else val)
                            for sub_k, val in conf[k].items() if sub_k
                            in layer_keys}
            else:  # output or layers
                model[k] = [{sub_k: (get_function(val) if sub_k
                                     in function_keys else val)
                            for sub_k, val in layer.items() if sub_k
                             in layer_keys}
                            for layer in conf[k]]
        else:
            raise KeyError(f"Invalid key '{k}' in configuration file.")
    if model['features'] is not None:
        model['input']['shape'] = len(model['features'])
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
        model['features'] = args.features
        model['input']['shape'] = len(model['features'])

    # Fill model output shape
    if model['output']['activation'] is None:
        model['output']['weights_initializer'] = he_initialisation
    return model


def fill_model_datasets(
        model: dict,
        dataset,
        training_rate: float,
        seed: int,
        target: str,
        features: list | None = None
        ) -> dict:
    """Split dataset and fill model with training and validation set

    Parameters:
      model (dict): Model parameters to fill with datasets
      dataset (pandas.DataFrame): The complete dataset
      training_rate (float): Ratio of the dataset to use for training
      seed (int): Seed for random operations to ensure reproducibility
      features (list, optional): List of feature column names to use.
                                 If empty, uses every column except target
      target (str): Name of the target column in the dataset``

    Returns:
      dict: Model parameters populated with training and validation datasets
    """
    df = pd.read_csv(dataset)
    if features is None:
        features = [col for col in df.columns
                    if col != target and col != 'id']
    filtered_df = pd.DataFrame(df[features + [target]])
    # Split before standardizing, and fit the statistics on the training set
    # only: standardizing the whole dataframe first would leak the validation
    # set's mean/std into training.
    train_df, test_df = split_dataset(filtered_df, ratio=training_rate,
                                      seed=seed)
    model['standardization'] = get_standardization_stats(train_df, features)
    model['data_train'] = standardize_df(train_df, features,
                                         model['standardization'])
    model['data_test'] = standardize_df(test_df, features,
                                        model['standardization'])
    model['input']['train_data'] = model['data_train'][features].to_numpy()
    model['input']['test_data'] = model['data_test'][features].to_numpy()
    model['output']['activation'] = softmax
    # Derived from the whole dataframe, before the split, so the order does
    # not depend on which rows land in the training set. Saved with the model
    # and reused at prediction time.
    model['classes'] = get_class_list(filtered_df, target)
    model['output']['shape'] = len(model['classes'])
    model['features'] = features
    model['input']['shape'] = len(features)
    model['target'] = target
    return model


def create_model(args, target: str, features: list | None = None) -> dict:
    """Create and initialize model parameters

    Parameters:
      args (argparse.Namespace): Parsed program arguments
      target (str): Name of the target column in the dataset
      features (list, optional): List of feature column names to use.
                                 If empty, uses model['input']['features'].
    """
    model = init_model_template()
    if features is not None:
        model['features'] = features
        model['input']['shape'] = len(features)
    if args.conf is not None:
        # args.conf is a path: fill_model_from_json needs an open file.
        with open(args.conf, 'r') as config_file:
            model = fill_model_from_json(model, config_file)
    model = fill_model_from_param(args, model)
    model = fill_model_datasets(model, args.dataset, args.train_ratio,
                                model['seed'], target, model['features'])
    # Set default loss function if not specified
    if model['loss'] is None:
        model['loss'] = FUNCTION_MAP['categoricalCrossentropy']
    if model['batch'] is not None:
        if 1 < model['batch'] < len(model['input']['train_data']):
            model['optimizer'] = 'mini-batch'
        elif model['batch'] >= len(model['input']['train_data']):
            model['batch'] = len(model['input']['train_data'])
            model['optimizer'] = 'batch'
        elif model['batch'] == 1:
            model['optimizer'] = 'stochastic'
    else:
        model['batch'] = len(model['input']['train_data'])
        model['optimizer'] = 'batch'

    # Set derivatives for each layer
    for layer in model['layers']:
        activation = layer['activation']
        if activation not in DERIVATIVE_MAP:
            name = getattr(activation, '__name__', activation)
            raise Exception(f"No derivative is implemented for the '{name}' "
                            "activation. Hidden layers must use sigmoid.")
        layer['derivative'] = DERIVATIVE_MAP[activation]
    return model


def load_model_from_json(filename: str) -> dict:
    """Load model parameters from a JSON file

    Parameters:
      filename (str) : Path to JSON file containing model parameters

    Returns:
      dict: Model parameters loaded from JSON file
    """
    model = init_model_template()
    with open(filename, 'r') as f:
        filled_model = fill_model_from_json(model, f)
    return filled_model
