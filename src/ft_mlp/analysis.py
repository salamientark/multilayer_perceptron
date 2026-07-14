import numpy as np
import pandas as pd
from .ft_math import ft_mean, ft_std, ft_min, ft_max, ft_q1, ft_q2, ft_q3, \
                     ft_variance, ft_skew, ft_kurtosis
from .preprocessing import get_numerical_features, remove_nan


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
    i = 0
    step = 4
    while i < len(features):
        print(f"{'':<12}", end="")
        j = i
        while j < step and j < len(features):
            print(f"{str(features[j])[:15]:>20}", end="")
            j += 1
        print()
        j = i
        print(f"{'Count':<12}", end="")
        while j < step and j < len(features):
            print(f"{counts[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'Mean':<12}", end="")
        while j < step and j < len(features):
            print(f"{means[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'Std':<12}", end="")
        while j < step and j < len(features):
            print(f"{stds[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'Min':<12}", end="")
        while j < step and j < len(features):
            print(f"{mins[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'25%':<12}", end="")
        while j < step and j < len(features):
            print(f"{q1s[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'50%':<12}", end="")
        while j < step and j < len(features):
            print(f"{q2s[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'75%':<12}", end="")
        while j < step and j < len(features):
            print(f"{q3s[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'Max':<12}", end="")
        while j < step and j < len(features):
            print(f"{maxs[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'Variance':<12}", end="")
        while j < step and j < len(features):
            print(f"{variances[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'Skewness':<12}", end="")
        while j < step and j < len(features):
            print(f"{skewness[j]:>20.6f}", end="")
            j += 1
        print()
        j = i
        print(f"{'Kurtosis':<12}", end="")
        while j < step and j < len(features):
            print(f"{kurtosis[j]:>20.6f}", end="")
            j += 1
        print()
        print()
        i += 4
        step += 4
    return


def ft_describe(df: pd.DataFrame, exclude: list = []):
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
