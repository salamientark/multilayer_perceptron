import numpy as np
import pandas as pd
from .ft_math import ft_mean, ft_std, ft_min, ft_max, ft_q1, ft_q2, ft_q3, \
                     ft_variance, ft_skew, ft_kurtosis
from .preprocessing import get_numerical_features, remove_nan

COLUMNS_PER_CHUNK = 4


def print_describe_result(
        features: list,
        counts: list,
        means: list,
        stds: list,
        mins: list,
        q1s: list,
        q2s: list,
        q3s: list,
        maxs: list,
        variances: list,
        skewness: list,
        kurtosis: list
        ):
    """Print the result of the description.

    Parameters:
      features (list): List of features.
      counts (list): List of counts.
      means (list): List of means.
      stds (list): List of standard deviations.
      mins (list): List of minimums.
      q1s (list): List of first quartiles.
      q2s (list): List of medians.
      q3s (list): List of third quartiles.
      maxs (list): List of maximums.
      variances (list): List of variances.
      skewness (list): List of skewnesses.
      kurtosis (list): List of kurtoses.
    """
    rows = [
        ('Count', counts),
        ('Mean', means),
        ('Std', stds),
        ('Min', mins),
        ('25%', q1s),
        ('50%', q2s),
        ('75%', q3s),
        ('Max', maxs),
        ('Variance', variances),
        ('Skewness', skewness),
        ('Kurtosis', kurtosis),
        ]
    # The table is printed in chunks of COLUMNS_PER_CHUNK features so that a
    # 30-feature dataset stays readable on a terminal.
    for start in range(0, len(features), COLUMNS_PER_CHUNK):
        chunk = range(start, min(start + COLUMNS_PER_CHUNK, len(features)))
        print(f"{'':<12}", end="")
        for j in chunk:
            print(f"{str(features[j])[:15]:>20}", end="")
        print()
        for label, values in rows:
            print(f"{label:<12}", end="")
            for j in chunk:
                print(f"{values[j]:>20.6f}", end="")
            print()
        print()
    return


def ft_describe(df: pd.DataFrame, exclude: list | None = None):
    """Describe the dataset given as parameter.

    Parameters:
      df (pd.Dataframe): Dataframe to describe
      exclude (list) (optionnal): List of column to exclude
    """
    features = get_numerical_features(df, exclude=exclude)
    counts, means, stds, mins, q1s, q2s = [], [], [], [], [], []
    q3s, maxs = [], []
    skewness, kurtosis, variance = [], [], []
    for feature in features:
        col = remove_nan(df[feature])
        size = len(col)
        counts.append(size)
        means.append(ft_mean(col, count=size))
        stds.append(ft_std(col, count=size))
        mins.append(ft_min(col))
        q1s.append(ft_q1(col, count=size))
        q2s.append(ft_q2(col, count=size))
        q3s.append(ft_q3(col, count=size))
        maxs.append(ft_max(col))
        variance.append(ft_variance(col, mean=means[-1], count=size))
        skewness.append(ft_skew(
            col, mean=means[-1], std=stds[-1], count=size))
        kurtosis.append(ft_kurtosis(
            col, mean=means[-1], std=stds[-1], count=size))
    print_describe_result(features, counts, means, stds, mins, q1s, q2s, q3s,
                          maxs, variance, skewness, kurtosis)


def ft_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return dataframe shape

    Parameters:
      df (pandas.DataFrame): Dataframe

    Return:
      tuple[int, int] : (row_number, col_number)
    """
    return (len(df), len(df.columns))


def correlation_coefficient(
        x: np.ndarray,
        y: np.ndarray,
        count: int | None = None
        ) -> float:
    """Calculate the correlation coefficient between two features

    Parameters:
      x (np.ndarray): First feature
      y (np.ndarray): Second feature
      count (int) (optional): number of observation
    """
    c = count if count is not None else len(x)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    numerator = c * np.dot(x, y) - x_sum * y_sum
    denominator = ((c * np.dot(x, x) - x_sum ** 2) *
                   (c * np.dot(y, y) - y_sum ** 2)) ** 0.5
    return numerator / denominator if denominator != 0 else 0


def correlation_matrix(df: pd.DataFrame) -> np.ndarray:
    """Calculate the correlation matrix for the dataframe

    Parameters:
      df (pd.DataFrame): Dataframe containing numerical features
    """
    row_nbr, col_nbr = ft_shape(df)
    corr_matrix = np.zeros((col_nbr, col_nbr))
    for x in range(col_nbr):
        x_feature = df.iloc[:, x]
        for y in range(x + 1):
            y_feature = df.iloc[:, y]
            if x == y:
                corr_matrix[x][y] = 1
                continue
            corr_coef = correlation_coefficient(x_feature, y_feature,
                                                count=row_nbr)
            corr_matrix[x][y] = corr_coef
            corr_matrix[y][x] = corr_coef
    return corr_matrix
