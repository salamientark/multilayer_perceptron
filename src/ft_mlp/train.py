import argparse as ap
import numpy as np
import ft_mlp as ft_mlp
import matplotlib.pyplot as plt
from math import ceil
# Imported directly: `ft_mlp.predict` resolves to the ft_mlp.predict module
# rather than the re-exported function whenever that module is imported.
from ft_mlp.network_layers import predict as nn_predict


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
DEFAULT_SHAPE = [24, 24]
DEFAULT_EPOCH = 84
DEFAULT_ALPHA = 0.1
DEFAULT_BATCH = 32
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_WEIGHTS_FILE = 'weights.npz'
DEFAULT_MODEL_FILE = 'trained_model.json'


def parse_args():
    """Parse program argument"""
    # Init parser
    parser = ap.ArgumentParser(prog="train.py",
                               description="Train a multilayer perceptron "
                                           "model.",
                               epilog=".... . .-.. .-.. --- / .-- --- .-. "
                                      ".-.. -.. -.-.--")

    # Create exclusion group for shape and layer
    shape_group = parser.add_mutually_exclusive_group()
    # Add parser argument
    shape_group.add_argument("--shape", type=int, nargs='+',
                             help="Define the number of neurons for each "
                                  f"hidden layer. (default: {DEFAULT_SHAPE})")
    shape_group.add_argument("--layer", type=int,
                             help="Define the number of hidden layers (use "
                                  "with --neurons).")
    shape_group.add_argument("--conf", type=str,
                             help="Model configuration file.")
    parser.add_argument("--features", choices=FEATURES, nargs='*',
                        help="List of features to use.")
    parser.add_argument("--neurons", type=int, required=False,
                        help="Define a constant number of neurons for all "
                             "hidden layers (use with --layer).")
    parser.add_argument("--loss", choices=['categoricalCrossentropy'],
                        help="Loss function to use "
                             "(default: categoricalCrossentropy)")
    parser.add_argument("--epoch", "-e", type=int,
                        help="Number of iteration of the trainig. "
                             f"(default: {DEFAULT_EPOCH})")
    parser.add_argument("--learning_rate", "-a", type=float, dest="alpha",
                        help="Learning rate of the algorithm. "
                             f"(default: {DEFAULT_ALPHA})")
    parser.add_argument("--batch", "-b", type=int, required=False,
                        help="Batch size if mini-batch gradient descent is "
                             f"used. (default: {DEFAULT_BATCH})")
    parser.add_argument("--seed", "-s", type=int,
                        help="Seed to make model reproducible "
                             f"(default: {DEFAULT_SEED})")
    parser.add_argument("--train_ratio", "-tr", type=float,
                        default=DEFAULT_TRAIN_RATIO,
                        help="The part of the dataset used as the training "
                             "set. (validation set ratio = 1 - train_ratio)")
    parser.add_argument("--outfile", "-of", type=str,
                        default=DEFAULT_WEIGHTS_FILE,
                        help="Weight result file (npz). "
                             f"(default: {DEFAULT_WEIGHTS_FILE})")
    parser.add_argument("--model_outfile", "-mo", type=str,
                        default=DEFAULT_MODEL_FILE,
                        help="Model topology result file (json). "
                             f"(default: {DEFAULT_MODEL_FILE})")
    parser.add_argument("dataset", type=str,
                        help="Training dataset.")
    # Get args
    args = parser.parse_args()
    return args


def validate_args(args):
    """Check args to check validity

    Parameters:
      args: program parameters
    """
    # Defaults are applied here rather than in argparse: a non-None value
    # would override the --conf file in fill_model_from_param().
    if args.conf is None:
        if args.seed is None:
            args.seed = DEFAULT_SEED
        if args.epoch is None:
            args.epoch = DEFAULT_EPOCH
        if args.train_ratio is None:
            args.train_ratio = DEFAULT_TRAIN_RATIO
        if args.alpha is None:
            args.alpha = DEFAULT_ALPHA
        if args.batch is None:
            args.batch = DEFAULT_BATCH
    if (args.layer is None) != (args.neurons is None):
        raise Exception("--layer and --neurons MUST be used together")
    if args.shape is None and args.layer is not None:
        args.shape = [args.neurons] * args.layer
    if args.shape is None and args.conf is None:
        args.shape = list(DEFAULT_SHAPE)
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
    if args.batch is not None and args.batch <= 0:
        raise Exception("Batch size must be a positive integer.")


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
    if model['optimizer'] is None or model['optimizer'] not in \
            ['mini-batch', 'stochastic', 'batch']:
        raise Exception("Optimizer must be mini-batch or stochastic.")
    if model['input']['shape'] is None or model['input']['shape'] <= 0:
        raise Exception("Input layer must have a positive number of neurons.")
    if (model['features'] is None
            or len(model['features']) == 0):
        raise Exception("Input layer must have at least one feature.")
    if model['output']['shape'] is None or model['output']['shape'] <= 0:
        raise Exception("Output layer must have a positive number of neurons.")
    if model['classes'] is None or len(model['classes']) < 2:
        raise Exception("The target column must contain at least two classes.")
    if len(model['classes']) != model['output']['shape']:
        raise Exception("Output layer shape must match the number of classes.")
    if (model['output']['activation'] is None or model['output']['activation']
            is not ft_mlp.softmax):
        raise Exception("Output layer activation must be softmax.")
    if (model['output']['weights_initializer'] is None or
            model['output']['weights_initializer']
            is not ft_mlp.he_initialisation):
        raise Exception("Output layer weight initialization must be heUniform "
                        "initialization.")
    for layer in model['layers']:
        if layer['shape'] is None or layer['shape'] <= 0:
            raise Exception("All hidden layers must have a positive number of"
                            " neurons.")
        if (layer['activation'] is None
                or layer['activation'] is not ft_mlp.sigmoid):
            raise Exception("All hidden layers must use the sigmoid "
                            "activation function.")
        if (layer['weights_initializer'] is None
            or layer['weights_initializer']
                is not ft_mlp.he_initialisation):
            raise Exception("All hidden layers must use HeUniform "
                            "initialization.")


def init_model(model: dict) -> dict:
    """Initialize model weights and bias

    Parameters:
      model (dict): Model parameters to initialize

    Returns:
      dict: Model with initialized weights and bias
    """
    seed = model['seed']
    # Each layer gets its own seed: the same seed for every layer would give
    # identically initialized layers whenever two layers have the same shape.
    for i, layer in enumerate(model['layers']):
        layer['gradients'] = {}
        fan_in = (model['input']['shape'] if i == 0
                  else model['layers'][i - 1]['shape'])
        layer['weights'], layer['bias'] = layer['weights_initializer'](
                fan_in, layer['shape'], seed + i)
    model['output']['weights'], model['output']['bias'] = \
        model['output']['weights_initializer'](
                model['layers'][-1]['shape'], model['output']['shape'],
                seed + len(model['layers'])
            )
    model['train_truth'] = ft_mlp.one_encode(model['data_train'],
                                             TARGET, model['classes'])
    model['test_truth'] = ft_mlp.one_encode(model['data_test'],
                                            TARGET, model['classes'])
    # One mean loss/accuracy value per epoch, for both sets, so that the
    # training and validation curves are directly comparable.
    model['train_loss'] = np.zeros(model['epoch'])
    model['test_loss'] = np.zeros(model['epoch'])
    model['train_acc'] = np.zeros(model['epoch'])
    model['test_acc'] = np.zeros(model['epoch'])
    return model


def feed_forward(model: dict, inputs: np.ndarray) -> list[np.ndarray]:
    """Perform feed forward pass in the mlp

    Result contain each layer activation result instead of just prediction.
    Used in model training for backpropagation

    Parameters:
      model (dict): Model parameters to use for feed forward pass
      inputs (np.ndarray): Input data to use for feed forward pass

    Returns:
      list[np.ndarray]: List of each layer activation result
    """
    layer_input = inputs
    result = []
    for layer in model['layers']:
        result.append(ft_mlp.hidden_layer(
                layer_input, layer['weights'],
                layer['bias'],
                activation=layer['activation']))
        layer_input = result[-1]
    result.append(ft_mlp.hidden_layer(
                layer_input, model['output']['weights'],
                model['output']['bias'],
                activation=model['output']['activation']))
    return result


def backpropagation(
        model: dict,
        inputs: np.ndarray,
        results: list[np.ndarray],
        truth: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Perform backpropagation to compute gradients

    Perform backpropagation to compute gradients for each layer in training
    process.

    Parameters:
      model (dict): Model parameters to use for backpropagation
      inputs (np.ndarray): Input data used in feed forward pass
      results (list[np.ndarray]): List of each layer activation result
      truth (np.ndarray): True output data

    Returns:
      list[tuple[np.ndarray]]: List of gradients for each layer (weights, bias)
    """
    gradients = []
    predictions = results[-1]
    # Partial derivative (Crossentropy/softmax), averaged over the batch so
    # that the effective learning rate does not scale with the batch size.
    # Every downstream gradient inherits this 1/m factor.
    gradient = (predictions - truth) / len(truth)
    gradient_weights = results[-2].T @ gradient
    gradient_bias = np.sum(gradient, axis=0)
    gradients.append((gradient_weights, gradient_bias))

    weights = model['output']['weights']
    for i in range(len(model['layers']) - 1, -1, -1):
        gradient = gradient @ weights.T
        gradient *= model['layers'][i]['derivative'](results[i])
        if i == 0:
            gradients_weights = inputs.T @ gradient
        else:
            gradients_weights = results[i - 1].T @ gradient
        gradient_bias = np.sum(gradient, axis=0)
        gradients.insert(0, (gradients_weights, gradient_bias))
        weights = model['layers'][i]['weights']

    return gradients


def update_weights(
        model: dict,
        gradients: list[tuple[np.ndarray, np.ndarray]]
        ) -> None:
    """Update model weights after training

    Parameters:
      model (dict): Model parameters to update
      gradients (list[tuple[np.ndarray]]): List of gradients for each layer
    """
    alpha = model['alpha']
    for i, layer in enumerate(model['layers']):
        layer['weights'] -= alpha * gradients[i][0]
        layer['bias'] -= alpha * gradients[i][1]

    model['output']['weights'] -= alpha * gradients[-1][0]
    model['output']['bias'] -= alpha * gradients[-1][1]


def print_training_state(epoch: int, model: dict):
    """Print training state for given epoch

    Parameters:
      epoch (int): Current epoch number
      model (dict): Model parameters to use for printing
    """
    epoch_len = len(str(model['epoch']))
    total_epoch = model['epoch']
    train_loss_mean = model['train_loss'][epoch]
    test_loss_mean = model['test_loss'][epoch]
    print(f"Epoch {epoch + 1:0{epoch_len}d}/{total_epoch:0{epoch_len}d} - "
          f"loss: {train_loss_mean:.6f} - "
          f"val_loss: {test_loss_mean:.6f}")


def train(model: dict):
    """Perform the training of the model

    Parameters:
      model (dict): Model parameters to train
    """
    # Print param
    print("data_train shape:", model['data_train'].shape)
    print("data_validation shape:", model['data_test'].shape)
    loss = model['loss']
    features = model['features']
    for i in range(model['epoch']):
        # Batch handling
        batch_size = model['batch']
        # Seed per epoch: a different shuffle each epoch, but the same
        # sequence of shuffles across runs with the same --seed.
        batch_indexes = ft_mlp.get_random_batch_indexes(
                model['data_train'].shape[0], seed=model['seed'] + i)
        batch = batch_indexes[:batch_size]
        total_batch = ceil(len(model['input']['train_data']) / batch_size)
        last_index = batch_size
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        # Train part
        while batch.size > 0:
            # Init
            inputs = model['input']['train_data'][batch]
            truth = model['train_truth'][batch]

            # Train step
            result = feed_forward(model, inputs)
            epoch_train_loss += ft_mlp.calculate_loss_mean(
                    result[-1],
                    truth,
                    loss)
            epoch_train_acc += ft_mlp.calculate_accuracy(
                    result[-1],
                    truth)
            gradients = backpropagation(model, inputs, result, truth)
            update_weights(model, gradients)

            # Next batch
            batch = batch_indexes[last_index:last_index + batch_size]
            last_index += batch_size

        # End of batch
        model['train_loss'][i] = epoch_train_loss / total_batch
        model['train_acc'][i] = epoch_train_acc / total_batch

        # Validation
        test_predictions = nn_predict(model, model['data_test'][features])
        test_truth = model['test_truth']
        model['test_loss'][i] = ft_mlp.calculate_loss_mean(
                test_predictions,
                test_truth,
                loss)
        model['test_acc'][i] = ft_mlp.calculate_accuracy(
                test_predictions,
                test_truth)

        print_training_state(i, model)


def plot_loss_and_accuracy_curves(model: dict):
    """Plot loss curve  and accuracy for training and validation set

    Parameters:
      model (dict): Model parameters to use for plotting
    """
    train_loss = model['train_loss']
    test_loss = model['test_loss']
    train_acc = model['train_acc']
    test_acc = model['test_acc']

    # Create a figure and a 1x2 grid of subplots (1 row, 2 columns)
    # 'figsize' is optional, but good for controlling the size of the figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot on the first subplot (left one)
    axes[0].plot(train_loss, color='blue', label='Training loss')
    axes[0].plot(test_loss, color='orange', linestyle='-', label='Test loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('loss')
    axes[0].set_xlim(0, model['epoch'] - 1)
    axes[0].legend()

    # Plot on the second subplot (right one)
    axes[1].plot(train_acc, color='blue', label='Training accuracy')
    axes[1].plot(test_acc, color='orange', linestyle='-',
                 label='Test accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].set_xlim(0, model['epoch'] - 1)
    axes[1].legend()

    manager = plt.get_current_fig_manager()
    if manager is not None:
        manager.full_screen_toggle()
    plt.show()


def main(args: ap.Namespace):
    """Train the model

    Parameters:
    args: argparse.Namespace
        Parsed program arguments
    """
    features = args.features if args.features is not None else FEATURES
    model = ft_mlp.create_model(args, TARGET, features)
    check_model(model)  # Validate model inputs
    init_model(model)  # Init model weights and bias
    train(model)
    ft_mlp.save_weights(args.outfile, model)
    ft_mlp.save_model(args.model_outfile, model)

    plot_loss_and_accuracy_curves(model)
    return


def cli():
    """Entry point for the command line."""
    args = parse_args()
    try:
        validate_args(args)
        main(args)
    except Exception as e:
        print(f"{ft_mlp.RED}Error{ft_mlp.RESET}: {e}")


if __name__ == "__main__":
    cli()
