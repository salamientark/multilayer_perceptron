import sys
import pandas as pd
import ft_datatools as ftdt


# COLORS
RED = '\033[91m'
RESET = '\033[0m'


def main(ac: int, av: list):
    """Give basic detailles on data

    Parameters:
      ac (int) : Number of parameters
      av (list) : List of parameters
    """
    try:
        if ac != 2:
            raise Exception("Usage: python analyzer.py <dataset.csv>")
        df = pd.read_csv(av[1], header=None)
        print(df)
        ftdt.ft_describe(df, exclude=['0'])
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
