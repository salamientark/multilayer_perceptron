import numpy as np
import pandas as pd
import ft_math as ftm


###############################################################################
#                               DATA ANALYSIS                                 #
###############################################################################
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
    print_describe_result(features, counts, means, stds, mins, q1s, q2s, q3s,
                          maxs, variance, skewness, kurtosis)


###############################################################################
#                              DATA TREATMENT                                 #
###############################################################################
def select_columns(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Select only the specified columns from the dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      features (list): List of numerical features name.

    Returns:
      pd.DataFrame: Dataframe with only the specified columns.
    """
    return df[features]


def get_numerical_features(
        df: pd.DataFrame,
        exclude: list = []
        ) -> list:
    """Get the numerical features name only from a dataframe.

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


def remove_nan(col: np.ndarray) -> np.ndarray:
    """Filter a column to keep only numerical values.

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
    f = func if func is not None else ftm.ft_mean
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
    """Classify the dataframe into multiple dataframes based on the target

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
    m = mean if mean is not None else ftm.ft_mean(array)
    s = std if std is not None else ftm.ft_std(array)
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
        std = ftm.ft_std(col_data)
        if std == 0:
            standardized_df[col] = 0
        else:
            standardized_df[col] = standardize_array(col_data, std=std)
    return standardized_df


###############################################################################
#                                 ALGORITHM                                   #
###############################################################################
def init_thetas(classes: list, feature_nbr: int) -> dict:
    """Initialize thetas dictionary with zeros

    Parameters:
    classes (list): List of class names
    feature_nbr (int): Number of features
    """
    thetas = {}
    for elem in classes:
        thetas[elem] = np.zeros(feature_nbr)
    return thetas


def unstandardized_thetas(
        thetas: dict,
        df: pd.DataFrame,
        features: list
        ) -> dict:
    """Convert standardized thetas to unstandardized thetas

    Parameters:
      thetas (dict): Standardized thetas
      df (pd.DataFrame): Dataframe with original data
      features (list): List of features names

    Returns:
      dict: Unstandardized thetas
    """
    means = {feature: ftm.ft_mean(df[feature]) for feature in features}
    std = {feature: ftm.ft_std(df[feature], mean=means[feature])
           for feature in features}
    unstandardized = {
        cls: [
            theta[0] - sum(
                (theta[i] * means[features[i - 1]]) / std[features[i - 1]]
                for i in range(1, len(theta))
            )
        ] + [
            theta[i] / std[features[i - 1]]
            for i in range(1, len(theta))
        ]
        for cls, theta in thetas.items()
    }
    return unstandardized


def save_thetas(thetas: dict, features: list) -> None:
    """Save thetas to a file

    Parameters:
      thetas (dict): Thetas to save
      features (list): List of features names
    """
    with open("thetas.csv", "w") as f:
        f.write("Class,Bias," + ",".join(features) + "\n")
        for cls, theta in thetas.items():
            f.write(cls + "," + ",".join([str(t) for t in theta]) + "\n")


def score_function(thetas: np.ndarray, features: np.ndarray) -> float:
    """Calculate score function for one sample
    thetas and features has to be the same size

    Parameters:
      thetas (np.ndarray): Thetas values
      features (np.ndarray): Features values

    Returns:
      np.ndarray: Score value
    """
    tmp_theta = thetas
    if thetas.shape != features.shape:
        tmp_theta = thetas.reshape(-1, 1)
    return np.dot(features, tmp_theta)


def sigmoid(
        thetas: np.ndarray,
        features: np.ndarray,
        value: float | None = None
        ) -> float:
    """Calculate sigmoid function for one sample

    Parameters:
      thetas (np.ndarray): Thetas values for score function
      features (np.ndarray): Features values for score function
      value (float | None): If value is given, calculate sigmoid for this
                       value instead of score function

    Returns:
      np.ndarray: Sigmoid value
    """
    val = value if value is not None else score_function(thetas, features)
    return 1 / (1 + np.exp(-val))


def batch_gradient_descent(thetas: np.ndarray,
                           features: pd.DataFrame,
                           target: np.ndarray,
                           alpha: float,
                           hypothesis=score_function
                           ) -> np.ndarray:
    """Calculate new thetas values using batch gradient descent

    Parameters:
      thetas (np.ndarray): Current thetas values
      features (pd.DataFrame): Features values
      target (np.ndarray): Target values (0 or 1)
      alpha (float): Learning rate
      hypothesis (function) (optionnal): Hypothesis function to use if not
                                         provided score_function will be used

    Returns:
      float: New theta value
    """
    new_thetas = thetas.copy()
    prediction = np.array([hypothesis(thetas, row)
                          for row in features.to_numpy()])
    errors = prediction - target
    gradient = np.dot(errors, features.to_numpy()) / len(features)
    new_thetas -= alpha * gradient
    return new_thetas


def stochastic_gradient_descent(thetas: np.ndarray,
                                features: pd.DataFrame,
                                target: np.ndarray,
                                alpha: float,
                                hypothesis=score_function
                                ) -> np.ndarray:
    """Calculate new thetas values using stochastic gradient descent

    Parameters:
      thetas (np.ndarray): Current thetas values
      features (pd.DataFrame): Features values
      target (np.ndarray): Target values (0 or 1)
      alpha (float): Learning rate
      hypothesis (function) (optionnal): Hypothesis function to use if not
                                         provided score_function will be used

    Returns:
      array: New theta value
    """

    # Shuffle dataset
    merged_df = features.copy()
    merged_df['target'] = target
    shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
    target = shuffled_df['target'].to_numpy()
    data = shuffled_df.drop(columns=['target']).to_numpy()
    new_thetas = thetas.copy()
    for data_row, target_row in zip(data, target):
        prediction = hypothesis(new_thetas, data_row)  # Vector
        errors = prediction - target_row  # Vector
        new_thetas -= alpha * errors * data_row  # Vector
    return new_thetas


def mini_batch_gradient_descent(thetas: np.ndarray,
                                features: pd.DataFrame,
                                target: np.ndarray,
                                alpha: float,
                                batch_size: int = 32,
                                hypothesis=score_function
                                ) -> np.ndarray:
    """Calculate new thetas values using mini batch gradient descent

    Parameters:
      thetas (np.ndarray): Current thetas values
      features (pd.DataFrame): Features values
      target (np.ndarray): Target values (0 or 1)
      alpha (float): Learning rate
      batch_size (int): Size of the mini batch
      hypothesis (function) (optionnal): Hypothesis function to use if not
                                         provided score_function will be used

    Returns:
      array: New theta value
    """
    # Shuffle dataset
    merged_df = features.copy()
    merged_df['target'] = target
    shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
    target = shuffled_df['target'].to_numpy()
    data = shuffled_df.drop(columns=['target']).to_numpy()
    new_thetas = thetas.copy()
    for start in range(0, len(data), batch_size):
        end = ftm.ft_min([start + batch_size, len(data)])
        data_batch = data[start:end]  # Matrix
        target_batch = target[start:end]  # Vector
        predictions = np.array([hypothesis(thetas, row) for row in data_batch])
        errors = predictions - target_batch
        gradient = np.dot(errors, data_batch) / len(data_batch)
        new_thetas -= alpha * gradient
    return new_thetas
