import argparse as ap
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


def main(args: ap.Namespace):
    """Make prediction using the loaded model on dataset

    Parameters:
    args: argparse.Namespace
        Parsed program arguments
    """
    model = ft_mlp.load_predict_model(args.model, args.weights, args.data,
                                      features=FEATURES)
    verify_model(model)

    print('Exit OK')
    return


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"{ft_mlp.RED}Error{ft_mlp.RESET}: {e}")
