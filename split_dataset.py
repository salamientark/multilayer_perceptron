import sys
import pandas as pd


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


def parser(ac: int, av: list) -> dict:
    """Verify program argument validity raise Exception on Error

    Parameters:
      ac (int): Program parameters len
      av (list): Program options
    """
    option = {'seed': 1, 'ratio': 0.8}
    if ac == 2:
        return option
    if ac == 4:
        if av[1].strip() == "--seed":
            if int(av[2]) <= 0:
                raise Exception("--seed argument must be a positive integer")
            option['seed'] = int(av[2])
        elif av[1].strip() == "--train-ratio":
            if float(av[2]) <= 0.0 or float(av[2]) >= 1.0:
                raise Exception("--train-ratio argument must be a float"
                                " ]0.0, 1.0[")
            option['ratio'] = float(av[2])
        else:
            raise Exception(f"Unkown option \"{av[2]}\"")
        return option
    if av[1].strip() != "--seed" and av[1].strip() != "--train-ratio":
        raise Exception(f"Unkown option \"{av[1]}\"")
    if av[1].strip() == av[3].strip():
        raise Exception("Realy!?")
    if av[1].strip() == "--seed":
        if int(av[2]) <= 0:
            raise Exception("--seed argument must be a positive integer")
        option['seed'] = int(av[2])
    if av[1].strip() == "--train-ratio":
        if float(av[2]) <= 0.0 or float(av[2]) >= 1.0:
            raise Exception("--train-ratio argument must be a float"
                            " ]0.0, 1.0[")
        option['ratio'] = float(av[2])
    if av[3].strip() == "--seed":
        if int(av[4]) <= 0:
            raise Exception("--seed argument must be a positive integer")
        option['seed'] = int(av[4])
    if av[3].strip() == "--train-ratio":
        if float(av[4]) <= 0.0 or float(av[4]) >= 1.0:
            raise Exception("--train-ratio argument must be a float"
                            " ]0.0, 1.0[")
        option['ratio'] = float(av[4])
    return option


def main(ac: int, av: list):
    """Split the dataset to get train and validation dataset

    Name
      split_dataset.py

    Usage:
      python split_dataset.py [OPTION] <dataset>

    Description:
      Split the dataset to get train and validation dataset

      --seed <int>        Random seed (default: 1)
      --train-ratio <float>  Ratio of training dataset (default: 0.8)
    """
    try:
        if ac % 2 != 0 or (ac < 2 and ac > 6):
            raise Exception("Usage: python split_dataset.py [OPTION] "
                            "<dataset>")
        # Get program options
        options = parser(ac, av)

        # Get Dataframe
        df = pd.read_csv(av[-1])
        df.columns = data_columns_names

        # Splitting
        train_set = df.sample(frac=options['ratio'],
                              random_state=options['seed']
                              ).reset_index(drop=True)
        validation_set = df[~df['id'].isin(train_set['id'])]

        # Wrinting to file
        train_set.to_csv("data_training.csv", index=False)
        validation_set.to_csv("data_validation.csv", index=False)
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
