import argparse as ap
import numpy as np
import ft_mlp as ft_mlp


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
OUTFILE = 'prediction.csv'


def parse_args():
    """Parse program argument"""
    parser = ap.ArgumentParser(
            prog="predict.py",
            description="Use a mlp model to make prediction on a given "
                        "dataset.",
            epilog=".... . .-.. .-.. --- / .-- --- .-. .-.. -.. -.-.--")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to the model file.")
    parser.add_argument("--weights", "-w",
                        type=str,
                        required=True, help="Path to the model weights file.")
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Path to the dataset file.")
    args = parser.parse_args()
    return args


def verify_model(model: dict):
    """Verify that the model is correctly loaded for prediction

    Verify that loaded model and weights are compatible for predicition.

    Parameters:
      model (dict) : Model structure
    """
    if model['data'] is None:
        raise ValueError("Model data is not correctly loaded.")
    if model['data'].shape[1] != model['input']['shape']:
        raise ValueError("Model data shape do not match model structure.")
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


def one_decoded(onecoded: np.ndarray, class_list: list) -> list:
    """Decode one-hot encoded array into class labels

    Used for prediction results decoding.

    Parameters:
    onecoded (np.ndarray): One-hot encoded array
    class_list (list): List of class labels

    Return:
    list: Decoded class labels
    """
    decoded = []
    indexes = ft_mlp.ft_argmax(onecoded)
    for i in indexes:
        decoded.append(class_list[i])
    return decoded


def main(args: ap.Namespace):
    """Make prediction using the loaded model on dataset

    Parameters:
    args: argparse.Namespace
        Parsed program arguments
    """
    # Init
    model = ft_mlp.load_predict_model(args.model, args.weights, args.data,
                                      features=FEATURES, target=TARGET)
    verify_model(model)

    # Prediction
    predictions = ft_mlp.predict(model, model['data'])
    decoded_predictions = one_decoded(predictions, model['truth_classes'])

    # Save predictions to file
    print(f"Saving result to file {ft_mlp.BLUE}{OUTFILE}{ft_mlp.RESET}"
          "... ", end="", flush=True)
    with open('prediction.csv', 'w') as f:
        f.write("Index,Prediction\n")
        for i, val in enumerate(decoded_predictions):
            f.write(f"{i},{val}\n")
    print(f"{ft_mlp.GREEN}Success{ft_mlp.RESET}")

    # Calculate BCE loss (Binary Cross Entropy)
    bce_loss = ft_mlp.binary_cross_entropy(predictions, model['truth'])
    bce_mean = ft_mlp.ft_mean(bce_loss)

    # Show prediction summary
    truth = model['raw_truth']
    matches = [pred == actual for pred, actual in zip(decoded_predictions,
                                                      truth)]
    valid_count = sum(matches)
    invalid_count = len(matches) - valid_count
    print(f"Valid predictions: {ft_mlp.GREEN}{valid_count}"
          f"{ft_mlp.RESET}\n"
          f"Invalid predictions: {ft_mlp.RED}{invalid_count}"
          f"{ft_mlp.RESET}\n"
          f"BCE loss: {ft_mlp.BLUE}{bce_mean}{ft_mlp.RESET}")
    return


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"{ft_mlp.RED}Error{ft_mlp.RESET}: {e}")
