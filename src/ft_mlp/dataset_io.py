"""Reading of the dataset csv files."""

import pandas as pd
from ft_mlp.dataset_schema import DATA_COLUMNS_NAMES, HEADER_MARKER


def read_dataset(dataset) -> pd.DataFrame:
    """Read a dataset csv whether or not it carries a header row.

    The raw data.csv shipped with the subject is headerless, while the files
    written by split_dataset carry a header. Sniffing which one we got lets
    every program accept both instead of failing with a pandas KeyError on
    the raw file.

    Parameters:
      dataset (str) : Path to the csv file to read

    Returns:
      pandas.DataFrame : The dataset, with named columns
    """
    if HEADER_MARKER in pd.read_csv(dataset, nrows=0).columns:
        return pd.read_csv(dataset)
    frame = pd.read_csv(dataset, header=None)
    if len(frame.columns) != len(DATA_COLUMNS_NAMES):
        raise Exception(f"{dataset}: headerless file has "
                        f"{len(frame.columns)} columns, expected "
                        f"{len(DATA_COLUMNS_NAMES)}. Is this the Wisconsin "
                        "breast cancer dataset?")
    # set_axis returns a new frame instead of mutating the one we just read.
    return frame.set_axis(DATA_COLUMNS_NAMES, axis=1)
