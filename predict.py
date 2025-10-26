import argparse as ap
import numpy as np
import ft_mlp as ft_mlp


def parse_args():
    """Parse program argument"""
    parser = ap.ArgumentParser(
            prog="predict.py",
            description="Use a mlp model to make prediction on a given "
                        "dataset.",
            epilog=".... . .-.. .-.. --- / .-- --- .-. .-.. -.. -.-.--")
    parser.add_argument("--model", "-m", type=ap.FileType('r'), required=True,
                        help="Path to the model file.")
    parser.add_argument("--weights", "-w",
                        type=ap.FileType('rb'),
                        required=True, help="Path to the model weights file.")
    parser.add_argument("--data", "-d", type=ap.FileType('r'), required=True,
                        help="Path to the dataset file.")
    args = parser.parse_args()
    return args


def load_weights_from_file(file) -> dict:
    """Load model weights from file

    Used for prediction.

    Parameters:
      file (str | file type) : File path or opened file IO buffer

    Returns:
      dict : Weights loaded from file
    """
    weights = np.load(file)
    return weights


def load_model_weights(model: dict, weights):
    """Load weights into model structure

    Used for prediction.

    Parameters:
      model (dict) : Model structure
      weights (np.lib.npyio.NpzFile) : Weights loaded from file
    """
    for i, layer in enumerate(model['layers']):
        weight_key = f'layer_{i}_weights'
        bias_key = f'layer_{i}_bias'
        layer['weights'] = weights[weight_key]
        layer['bias'] = weights[bias_key]
    model['output']['weights'] = weights['output_weights']
    model['output']['bias'] = weights['output_bias']


def load_predict_model(args: ap.Namespace) -> dict:
    """Load model and weights for prediction

    Parameters:
    args (argparse.Namespace): Parsed program arguments
    """
    model = ft_mlp.load_model_from_json(args.model)
    weights = load_weights_from_file(args.weights)
    load_model_weights(model, weights)

    # Load dataset
    model['data'] = np.loadtxt(args.data, skiprows=1)

    # Remove unneeded keys
    for layer in model['layers']:
        layer.pop('weights_initializer', None)
    model['output'].pop('weights_initializer', None)
    return model


def verify_model(model: dict):
    """Verify that the model is correctly loaded for prediction

    Verify that loaded model and weights are compatible for predicition.

    Parameters:
      model (dict) : Model structure
    """
    last_layer_index = len(model['layers']) - 1
    for i, layer in enumerate(model['layers']):
        if 'weights' not in layer or 'bias' not in layer:
            raise ValueError("Model weights are not correctly loaded.")
        # Hidden layers weights check
        # Layer shape must match weights shape
        if layer['weights'].shape[1] != layer['shape']:
            raise ValueError("Model weights shape do not match model "
                             "structure.")
        # Layer bias shape must match layer weights shape
        if layer['weights'].shape[1] != layer['bias'].shape[0]:
            raise ValueError("Model weights shape do not match model "
                             "structure.")
        # Inputs weight check
        if i == 0 and layer['weights'].shape[0] != model['input']['shape']:
            raise ValueError("Model weights shape do not match model "
                             "structure.")
        # Output weights check
        if i == last_layer_index:
            if layer['weights'].shape[1] != \
                    model['output']['weights'].shape[0]:
                raise ValueError("Model weights shape do not match model "
                                 "structure.")
            continue
        # layer weights output must match next layer weights input
        next_layer = model['layers'][i + 1]
        if layer['weights'].shape[1] != next_layer['weights'].shape[0]:
            raise ValueError("Model weights shape do not match model "
                             "structure.")


def main(args: ap.Namespace):
    """Make prediction using the loaded model on dataset

    Parameters:
    args: argparse.Namespace
        Parsed program arguments
    """
    # model = ft_mlp.load_model_from_json(args.model)
    # weights = load_weights_from_file(args.weights)
    # load_model_weights(model, weights)
    model = load_predict_model(args)
    verify_model(model)
    ft_mlp.print_model(model)

    print('Exit OK')
    return


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"{ft_mlp.RED}Error{ft_mlp.RESET}: {e}")
