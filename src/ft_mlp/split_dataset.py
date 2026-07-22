import argparse as ap
import sys
from ft_mlp import RED, RESET, GREEN, BLUE
from ft_mlp.dataset_io import read_dataset
from ft_mlp.dataset_schema import DATA_COLUMNS_NAMES
from ft_mlp.preprocessing import split_dataset


# Data columns name (kept as a module attribute for callers/tests)
data_columns_names = DATA_COLUMNS_NAMES


def parse_args():
    """Get program command line argument"""
    # Init parser
    parser = ap.ArgumentParser(prog="split_dataset.py",
                               description="Split the dataset into a training "
                                           "set and a validation set.",
                               epilog=">^-^<")

    # Add parser option
    # The default is a list: argparse does not apply nargs to a default, so a
    # "a,b" string default would be validated character by character.
    parser.add_argument("--outfile", "-o", type=str,
                        default=['data_training.csv', 'data_validation.csv'],
                        help="output dataset name", nargs='+')
    parser.add_argument("--seed", "-s", type=int, default=1,
                        help="Random seed for shufling.")
    parser.add_argument("--train-ratio", "-r", type=float, default=0.8,
                        help="ratio of the training set size.")
    parser.add_argument("dataset_path", help="Path to the input csv file.")

    # Parse program arguments
    args = parser.parse_args()

    # Custom validation part
    if not (0.0 < args.train_ratio < 1.0):
        parser.error("The --training argument must be a float between 0.0 "
                     "and 1.0")
    if not (0 < args.seed):
        parser.error("The --seed argument must be a positive integer")

    if len(args.outfile) == 1:
        outfiles = args.outfile[0].split(',')
        args.outfile = [file.strip() for file in outfiles
                        if file.strip() != ""]
        if not len(outfiles) == 2:
            parser.error("The --outfile argument must contain 2 filenames.")
        if outfiles[0] == outfiles[1]:
            parser.error("The --outfile argument must contain 2 different "
                         "filenames.")
    else:
        args.outfile = [file.strip() for file in args.outfile if
                        file.strip() != ""]
        if not len(args.outfile) == 2:
            parser.error("The --outfile argument must contain 2 filenames.")
        if args.outfile[0] == args.outfile[1]:
            parser.error("The --outfile argument must contain 2 different "
                         "filenames.")

    return args


def main(args):
    """Split the dataset to get train and validation dataset

    Name
      split_dataset.py

    Usage:
      python split_dataset.py [OPTION] <dataset>

    Description:
      Split the dataset to get train and validation dataset

      --seed <int>        Random seed (default: 1)
      --train-ratio <float>  Ratio of training dataset (default: 0.8)
      --outfiles <str,str> or <str> <str>   Output files
    """
    # Get Dataframe. read_dataset accepts the raw headerless data.csv as
    # well as an already-split file that carries a header, so re-splitting
    # an output of this program works too.
    df = read_dataset(args.dataset_path)

    # Splitting
    train_set, test_set = split_dataset(df, ratio=args.train_ratio,
                                        seed=args.seed)

    # Writing to file
    print(f"Saving training set to : {BLUE}{args.outfile[0]}{RESET} ... ",
          end="")
    train_set.to_csv(args.outfile[0], index=False)
    print(f"{GREEN}OK!{RESET}")

    print(f"Saving test set to     : {BLUE}{args.outfile[1]}{RESET} ... ",
          end="")
    test_set.to_csv(args.outfile[1], index=False)
    print(f"{GREEN}OK!{RESET}")


def cli():
    """Entry point for the command line."""
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
