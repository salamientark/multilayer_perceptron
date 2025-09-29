import argparse as ap
import pandas as pd


# Data columns name
data_columns_names = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concativity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_std",
    "texture_std",
    "perimeter_std",
    "area_std",
    "smoothness_std",
    "compactness_std",
    "concativity_std",
    "concave_points_std",
    "symmetry_std",
    "fractal_dimension_std",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concativity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


# COLORS
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def parse_args():
    """Get program command line argument"""
    # Init parser
    parser = ap.ArgumentParser(prog="split_dataset.py",
                               description="Split the dataset into a training "
                                           "set and a validation set.",
                               usage="usage: split_dataset.py [-h] [--seed "
                                     "SEED] [--train-ratio TRAIN_RATIO] "
                                     "<dataset_path> [--outfile OUTFILE "
                                     "[OUTFILE ...]]",
                               epilog=">^-^<")

    # Add parser option
    parser.add_argument("--seed", "-s", type=int, default=1,
                        help="Random seed for shufling.")
    parser.add_argument("--train-ratio", "-r", type=float, default=0.8,
                        help="ratio of the training set size.")
    parser.add_argument("dataset_path", help="Path to the input csv file.")
    parser.add_argument("--outfile", "-o", type=str,
                        default="data_training.csv,data_validation.csv",
                        help="output dataset name", nargs='+')

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
        outfiles = [file.strip() for file in outfiles if file.strip() != ""]
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
    try:
        # Get Dataframe
        df = pd.read_csv(args.dataset_path)
        df.columns = data_columns_names

        # Splitting
        train_set = df.sample(frac=args.train_ratio,
                              random_state=args.seed)
        validation_set = df.drop(train_set.index)

        # Wrinting to file
        train_set.to_csv(args.outfile[0], index=False)
        validation_set.to_csv(args.outfile[1], index=False)
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
