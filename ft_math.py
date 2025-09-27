import numpy as np
import pandas as pd


def ft_isnbr(elem) -> bool:
    """Check if an element is a number (int or float).

    Parameters:
      elem: Element to check.

    Returns:
      bool: True if the element is a number, False otherwise.
    """
    if isinstance(elem, (int, float)) and not pd.isna(elem):
        return True
    return False


def ft_mean(
        array: np.ndarray,
        count: int | None = None
        ) -> float:
    """Compute the mean of a list of numbers.

    Parameters:
      array (np.ndarray): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                              will be computed.

    Returns:
      float: Mean of the list of numbers.
    """
    c = count
    if c is None:
        c = len(array)
    mean = 0
    for i in range(c):
        mean += (1. / float(c)) * array[i]
    return mean


def ft_variance(
        array,
        mean: float | None = None,
        count: int | None = None
        ) -> float:
    """Compute the variance of a list of numbers.

    Parameters:
      array (list): List of numbers.
      mean (float) (optional): Mean of the list of numbers. If None, it will
                               be computed.
      count (int) (optionnal): Number of elements in the list. If None, it
                               will be computed.

    Returns:
      float: Variance of the list of numbers.
    """
    c = count
    m = mean
    if m is None:
        m = ft_mean(array)
    if c is None:
        c = len(array)
    var = 0
    for i in range(c):
        var += (1. / float(c)) * (array[i] - m) ** 2
    return var


def ft_std(
        array,
        mean: float | None = None,
        var: float | None = None,
        count: int | None = None
        ) -> float:
    """Compute the standard deviation of a list of numbers.

    Parameters:
      array (list): List of numbers.
      mean (float) (optional): Mean of the list of numbers. If None, it
                               will be computed.
      var (float) (optional): Variance of the list of numbers. If None,
                              it will be computed.
      count (int) (optionnal): Number of elements in the list. If None, it
                               will be computed.
    Returns:
      float: Standard deviation of the list of numbers.
    """
    c = count
    v = var
    if c is None:
        c = len(array)
    if v is None:
        v = ft_variance(array, count=c)
    return v ** 0.5


def ft_min(array) -> int | float:
    """Return the minimum of a list of numbers.

    Parameters:
      array (list): List of numbers.

    Returns:
      int | float: Minimum of the list of numbers.
    """
    sorted_array = sorted(array)
    return sorted_array[0]


def ft_max(array) -> int | float:
    """Return the maximum of a list of numbers.

    Parameters:
      array (list): List of numbers.

    Returns:
      int | float: Maximum of the list of numbers.
    """
    sorted_array = sorted(array)
    return sorted_array[-1]


def ft_q1(array, count: int | None = None) -> int | float:
    """Return the first quartile of a list of numbers.
    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                            will be computed.

    Returns:
      int | float: First quartile of the list of numbers.
    """
    c = count
    if c is None:
        c = len(array)
    sorted_array = sorted(array)
    return sorted_array[c // 4]


def ft_q2(array, count: int | None = None) -> int | float:
    """Return the second quartile (median) of a list of numbers.
    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                            will be computed.

    Returns:
      int | float: Second quartile of the list of numbers.
    """
    c = count
    if c is None:
        c = len(array)
    sorted_array = sorted(array)
    return sorted_array[c // 2]


def ft_q3(array, count: int | None = None) -> int | float:
    """Return the third quartile of a list of numbers.
    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                            will be computed.

    Returns:
      int | float: Third quartile of the list of numbers.
    """
    c = count
    if c is None:
        c = len(array)
    sorted_array = sorted(array)
    return sorted_array[(c // 4) * 3]


def ft_skew(
        array,
        count: int | None = None,
        mean: float | None = None,
        std: float | None = None
        ) -> float:
    """Compute the skewness of a list of numbers.
    Measures the asymmetry of the probability distribution.

    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                              will be computed.
      mean (float) (optional): Mean of the list of numbers. If None, it will
                               be computed.
      std (float) (optional): Standard deviation of the list of numbers. If
                              None, it will be computed.

    Returns:
      float: Skewness of the list of numbers.
    """
    c = count if count is not None else len(array)
    m = mean if mean is not None else ft_mean(array, count=c)
    st = std if std is not None else ft_std(array, mean=m, count=c)
    u = 0
    for i in range(c):
        u += (array[i] - m) ** 3
    u /= c
    return u / (st ** 3) if st != 0 else 0


def ft_kurtosis(
        array,
        count: int | None = None,
        mean: float | None = None,
        std: float | None = None
        ) -> float:
    """Compute the kurtosis of a list of numbers.
    Measures the flatness of the probability distribution.

    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                              will be computed.
      mean (float) (optional): Mean of the list of numbers. If None, it will
                               be computed.
      std (float) (optional): Standard deviation of the list of numbers. If
                              None, it will be computed.

    Returns:
      float: Kurosis of the list of numbers.
    """
    c = count if count is not None else len(array)
    m = mean if mean is not None else ft_mean(array, count=c)
    st = std if std is not None else ft_std(array, mean=m, count=c)
    u = 0
    for i in range(c):
        u += (array[i] - m) ** 4
    u /= c
    return u / (st ** 4) - 3 if st != 0 else 0
