import sys
import pandas as pd
import ft_math as ftm
import ft_datatools as ftdt


# COLORS
RED = '\033[91m'
RESET = '\033[0m'


def print_result(
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
            print(f"{features[j][:15]:>20}", end="")
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


def main(ac: int, av: list):
    """Describe the dataset given as parameter.

    Parameters:
      ac (int): Number of command line arguments.
      av (list): List of command line arguments.
    """
    try:
        if (ac != 2):
            raise Exception("Usage: describe.py <dataset_path>")
        df = pd.read_csv(av[1])  # Dataframe
        features = ftdt.get_numerical_features(df, exclude=['Index'])
        counts, means, stds, mins, q1s, q2s = [], [], [], [], [], []
        q3s, maxs = [], []
        skewness, kurtosis, variance = [], [], []
        for feature in features:
            col = ftdt.filter_col(df[feature])
            size = len(col)
            counts.append(size)
            means.append(ftm.ft_mean(col, count=size))
            stds.append(ftm.ft_std(col, count=size))
            mins.append(ftm.ft_min(col))
            q1s.append(ftm.ft_q1(col, count=size))
            q2s.append(ftm.ft_q2(col, count=size))
            q3s.append(ftm.ft_q3(col, count=size))
            maxs.append(ftm.ft_max(col))
            variance.append(ftm.ft_variance(col, mean=means[-1], count=size))
            skewness.append(ftm.ft_skew(
                col, mean=means[-1], std=stds[-1], count=size))
            kurtosis.append(ftm.ft_kurtosis(
                col, mean=means[-1], std=stds[-1], count=size))
        print_result(features, counts, means, stds, mins, q1s, q2s, q3s, maxs,
                     variance, skewness, kurtosis)
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
