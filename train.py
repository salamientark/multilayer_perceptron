import argparse as ap


# COLORS
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

    
def parse_args():
    """Parse program argument"""
    # Init parser
    parser = ap.ArgumentParser(prog="train.py",
                               description="Train a multilayer perceptron "
                                           "model.",
                               epilog=".... . .-.. .-.. --- / .-- --- .-. "
                                      ".-.. -.. -.-.--")

    # Create exclusion group for shape and layer
    shape_group = parser.add_mutually_exclusive_group(required=False)
    # Add parser argument
    shape_group.add_argument("--shape", "-s", type=int, nargs='+',
                             help="Define the number of neurons for each "
                                  "hidden layer.")
    shape_group.add_argument("--layer", type=int,
                             help="Define the number of hidden layers (use "
                                  "with --neurons).")
    parser.add_argument("--neurons", type=int, required=False,
                        help="Define a constant number of neurons for all "
                             "hidden layers (use with --layer).")
    parser.add_argument("--loss", choices=['categoricalCrossentropy'],
                        default="categoricalCrossentropy",
                        help="Loss function to use")
    # parser.add_argument("--hidden_activation", choices=['sigmoid', 'softmax'])
    parser.add_argument("--epoch", "-e", type=int, required=True,
                        help="Number of iteration of the trainig.")
    parser.add_argument("--learning_rate", "-a", type=float, required=True,
                        help="Learning rate of the algorithm.")
    parser.add_argument("--batch", "-b", type=int, required=False, default=8,
                        help="Batch size if mini-batch gradient descent is "
                             "used.")
    parser.add_argument("training_dataset", type=ap.FileType('r'),
                        help="Training dataset.")
    parser.add_argument("--outfile", "-of", type=ap.FileType('w'),
                        default="weights.csv", help="Weight result file.")
    # Get args
    args = parser.parse_args()

    # Custom error checking
    if args.layer is None != args.neurons is None:
        parser.error("--layer and --neurons MUST be used together")
    if args.shape is None and args.layer is None:
        parser.error("Either --shape or --layer and --neurons must be used")
    if args.shape is None and args.layer is not None:
        args.shape = [args.neurons] * args.layer
    if any(n <= 0 for n in args.shape):
        parser.error("All layer must have at least one neuron")
    if args.epoch <= 0:
        parser.error("Number of epoch must be > 0")
    if not (0 < args.learning_rate < 1):
        parser.error("Learning rate must be in the range (0, 1]")
    return args


def main(args):
    """Train the model"""
    try:
        df
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}")


if __name__ == "__main__":
    args = parse_args()

    main(args)
