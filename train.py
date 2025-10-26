import argparse as ap
import numpy as np
import ft_mlp as ft_mlp
import matplotlib.pyplot as plt
from math import ceil


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
                        default=0.8,
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
    if model['optimizer'] is None or model['optimizer'] not in \
            ['mini-batch', 'stochastic']:
        raise Exception("Optimizer must be mini-batch or stochastic.")
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
    if (model['output']['weights_initializer'] is None or
            model['output']['weights_initializer']
            is not ft_mlp.he_initialisation):
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
        if (layer['weights_initializer'] is None
            or layer['weights_initializer']
                is not ft_mlp.he_initialisation):
            raise Exception("All hidden layers must use zero weight "
                            "initialization.")


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
            layer['weights'], layer['bias'] = layer['weights_initializer'](
                    model['input']['shape'], layer['shape'], seed, data_inputs)
            continue
        layer['weights'], layer['bias'] = layer['weights_initializer'](
                model['layers'][i - 1]['shape'], layer['shape'], seed,
                data_inputs)
    model['output']['weights'], model['output']['bias'] = \
        model['output']['weights_initializer'](
                model['layers'][-1]['shape'], model['output']['shape'],
                seed, data_inputs
            )
    model['train_truth'] = ft_mlp.one_encoding(model['data_train'],
                                               TARGET)
    model['test_truth'] = ft_mlp.one_encoding(model['data_test'],
                                              TARGET)
    model['train_loss'] = np.zeros((model['epoch'], len(model['data_train'])))
    model['test_loss'] = np.zeros((model['epoch'], len(model['data_test'])))
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
    gradient = predictions - truth  # Partial derivative (Crossentropy/softmax)
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


def get_model_weights(model: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Get model weights as list of (weights, bias) tuple for each layer

    Used to save old weights for early stopping.

    Parameters:
      model (dict): Model parameters to get weights from

    Returns:
      list[tuple[np.ndarray]]: List of (weights, bias) tuple for each layer
    """
    weights = []
    for layer in model['layers']:
        weights.append((np.copy(layer['weights']),
                        np.copy(layer['bias'])))
    weights.append((np.copy(model['output']['weights']),
                    np.copy(model['output']['bias'])))
    return weights


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
    train_loss = model['train_loss']
    test_loss = model['test_loss']
    train_loss_mean = np.sum(train_loss[epoch]) / train_loss.shape[0]
    test_loss_mean = np.sum(test_loss[epoch]) / test_loss.shape[0]
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
    features = model['input']['features']
    for i in range(model['epoch']):
        # Batch handling
        batch_size = model['batch']
        batch_indexes = ft_mlp.get_random_batch_indexes(
                model['data_train'].shape[0])
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
        test_predictions = ft_mlp.predict(model, model['data_test'][features])
        test_truth = model['test_truth']
        model['test_loss'][i] = ft_mlp.calculate_loss(
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
    train_loss = np.sum(model['train_loss'], axis=1) \
        / len(model['data_train'])
    test_loss = np.sum(model['test_loss'], axis=1) \
        / len(model['data_test'])
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
    axes[0].legend()

    # Plot on the second subplot (right one)
    axes[1].plot(train_acc, color='blue', label='Training accuracy')
    axes[1].plot(test_acc, color='orange', linestyle='-',
                 label='Test accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()

    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()


def main(args: ap.Namespace):
    """Train the model

    Parameters:
    args: argparse.Namespace
        Parsed program arguments
    """
    model = ft_mlp.create_model(args, TARGET, FEATURES)
    check_model(model)  # Validate model inputs
    init_model(model)  # Init model weights and bias
    train(model)
    ft_mlp.save_weights("weights.npz", model)
    ft_mlp.save_model("trained_model.json", model)

    plot_loss_and_accuracy_curves(model)
    return


if __name__ == "__main__":
    args = parse_args()
    try:
        validate_args(args)
        main(args)
    except Exception as e:
        print(f"{ft_mlp.RED}Error{ft_mlp.RESET}: {e}")
