import argparse as ap
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
    parser.add_argument("--weights", "-w", type=ap.FileType('r'),
                        required=True, help="Path to the model weights file.")
    parser.add_argument("--data", "-d", type=ap.FileType('r'), required=True,
                        help="Path to the dataset file.")
    args = parser.parse_args()
    return args


def main(args: ap.Namespace):
    """Make prediction using the loaded model on dataset

    Parameters:
    args: argparse.Namespace
        Parsed program arguments
    """
    model = ft_mlp.load_model_from_json(args.model)
    ft_mlp.print_model(model)

    print('Exit OK')
    return


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"{ft_mlp.RED}Error{ft_mlp.RESET}: {e}")
