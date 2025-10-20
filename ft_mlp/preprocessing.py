import numpy as np
import pandas as pd
from .ft_math import ft_mean, ft_std


def select_columns(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Select o the specified columns from the dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      features (list): List of numerical features name.

    Returns:
      pd.DataFrame: Dataframe with o the specified columns.
    """
    return df[features]


def get_numerical_features(
        df: pd.DataFrame,
        exclude: list = []
        ) -> list:
    """Get the numerical features name o from a dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      exclude (list) (optional): List of columns to exclude.

    Returns:
      list: List of numerical features.
    """
    columns = df.columns.tolist()
    filtered_features = []
    for col in columns:
        if col in exclude:
            continue
        for elem in df[col].tolist():
            if elem is None or pd.isna(elem):
                continue
            if isinstance(elem, (int, float)):
                filtered_features.append(col)
            break
    return filtered_features


def get_class_list(df: pd.DataFrame, col: str) -> list:
    """Get the list of unique values in the specified column

    Parameters:
      df (pd.DataFrame): Dataframe.
      col (str): col column name.

    Returns:
      list: List of unique values in the col column. Empty if no class
    """
    try:
        target_col = df[col]
        result = []
        for elem in target_col:
            if pd.isna(elem):
                continue
            if elem not in result:
                result.append(elem)
        return result
    except KeyError as e:
        raise Exception(f"Column '{col}' not found in the dataframe.") from e


def convert_classes_to_nbr(class_name: str, data: pd.Series) -> pd.Series:
    """Convert class names to numerical values

    Parameters:
    class_name (str): Class name
    data (pd.Series): Series with class names

    Returns:
    pd.Series: Series with numerical values
    """
    converted_col = (data == class_name).astype(int)
    return converted_col.astype(int)


def one_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a column of class names to numerical values using one encoding

    Parameters:
      df (pd.DataFrame): Dataframe.
      col (str): Column name to convert.

    Returns:
      np.ndarray[int]: Array of numerical values.
    """
    class_list = get_class_list(df, col)
    one_encoded_val = {}
    for _, c in enumerate(class_list):
        one_encoded_val[c] = [int(val == c) for val in class_list]
    # l = df[col].to_numpy()
    # return np.array(list(map((lambda x: one_encoded_val[x]), l)))
    return np.array(list(map((lambda x: one_encoded_val[x]), df[col])))


def remove_nan(col: np.ndarray) -> np.ndarray:
    """Filter a column to keep o numerical values.

    Parameters:
      col (np.ndarray): Column to filter.

    Returns:
      np.ndarray: Filtered column.
    """
    col = np.array(col)
    return col[~pd.isna(col)]


def replace_nan(
        df: pd.DataFrame,
        columns: list = [],
        func=None
        ) -> pd.DataFrame:
    """Replace NaN values in a dataframe with the mean of the column

    Parameters:
      df (pd.DataFrame): Dataframe to process
      columns (list) (optionnal): List of columns to use of specified
    func (function) (optionnal): Function to use to compute the
            missing values. If None is provided the mean will be used.

    Returns:
      pd.DataFrame: Dataframe with NaN values replaced
    """
    new_df = df.copy()
    f = func if func is not None else ft_mean
    cols = columns if columns != [] else new_df.columns
    for column in cols:
        tmp_col = remove_nan(new_df[column].tolist())
        val = f(tmp_col)
        new_df[column] = new_df[column].fillna(val)
    return new_df


def remove_missing(df: pd.DataFrame, exclude: list[str] = []) -> pd.DataFrame:
    """Remove rows with missing values in the dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      exclude (list) (optional): List of columns to exclude.

    Returns:
      pd.DataFrame: Dataframe without missing values.
    """
    cleaned_df = df.copy()
    subset = [col for col in cleaned_df.columns if col not in exclude]
    cleaned_df.dropna(subset=subset, inplace=True)
    return cleaned_df


def classify(df: pd.DataFrame, target_col: str, features: list
             ) -> dict[str, pd.DataFrame]:
    """Classify the dataframe into tiple dataframes based on the target

    Parameters:
      df (pd.DataFrame): Dataframe.
      target (str): Target column name.
      features (list): List of numerical features name.

    Returns:
      dict[pd.DataFrame]: Dictionary of dataframes classified by class.
    """
    res = {}
    grouped = df.groupby(target_col)
    for key, group in grouped:
        res[key] = group[features].copy()
    return res


def standardize_array(array: np.ndarray,
                      mean: float | None = None,
                      std: float | None = None
                      ) -> np.ndarray:
    """Standardize a list of numerical values.

    Parameters:
    array (np.ndarray): List of numerical values.
    mean (float | None): Mean of the list. If None, it will be computed.
    std (float | None): Standard deviation of the list. If None, it will be
        computed.

    Returns:
    np.ndarray: Standardized list.
    """
    m = mean if mean is not None else ft_mean(array)
    s = std if std is not None else ft_std(array)
    standardized = array.copy()
    standardized = (standardized - m) / s if s != 0 else np.zeros(len(array))
    return standardized


def standardize_df(df: pd.DataFrame, columns: list = []) -> pd.DataFrame:
    """Standardize the specified columns of a dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      columns (list) (optional): List of columns to standardize. If empty,
                                 all numerical columns will be standardized.

    Returns:
      pd.DataFrame: Dataframe with standardized columns.
    """
    standardized_df = df.copy()
    if columns is None or not columns:
        columns = get_numerical_features(standardized_df)
    for col in columns:
        col_data = np.array(standardized_df[col].values)
        std = ft_std(col_data)
        if std == 0:
            standardized_df[col] = 0
        else:
            standardized_df[col] = standardize_array(col_data, std=std)
    return standardized_df


def split_dataset(df: pd.DataFrame,
                  ratio: float = 0.8,
                  seed: int = 1
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into tiple dataframes according to the given ratio

    Parameters:
    df (pd.DataFrame): Dataframe to split
    ratio (float): Ratio for split
    seed (int): random seed

    Returns:
      tuple[pandas.DataFrame, pandas.DataFrame): Tuple of dataframes
    """
    x = df.sample(frac=ratio, random_state=seed)
    return (x, df.drop(x.index))
