import numpy as np
import pandas as pd
import ft_math as ftm
from random import seed, randrange
from sys import maxsize


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


def ft_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return dataframe shape

    Parameters:
      df (pandas.DataFrame): Dataframe

    Return:
      tuple[int, int] : (row_number, col_number)
    """
    return (len(df), len(df.columns))


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
    for i, c in enumerate(class_list):
        one_encoded_val[c] = [int(val == c) for val in class_list]
    return df[col].map(one_encoded_val)


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


def split_dataset(df: pd.DataFrame,
                  ratio: float = 0.8,
                  seed: int = 1
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into multiple dataframes according to the given ratio

    Parameters:
    df (pd.DataFrame): Dataframe to split
    ratio (float): Ratio for split
    seed (int): random seed

    Returns:
      tuple[pandas.DataFrame, pandas.DataFrame): Tuple of dataframes
    """
    x = df.sample(frac=ratio, random_state=seed)
    return (x, df.drop(x.index))


###############################################################################
#                                 ALGORITHM                                   #
###############################################################################
def get_random_seed() -> int:
    """Generate a random positive int as seed

    Returns:
      int: Generated seed
    """
    seed()
    return randrange(1, maxsize)


def init_thetas(classes: list, feature_nbr: int) -> dict:
    """Initialize thetas dictionary with zeros

    Parameters:
      classes (list): List of class names
      feature_nbr (int): Number of features

    Returns:
      (dict): theta parameters for each class
    """
    thetas = {cls: np.zeros(feature_nbr) for cls in classes}
    # for elem in classes:
    #     thetas[elem] = np.zeros(feature_nbr)
    return thetas


def init_weights_zero(features: int, output: int) -> tuple[np.ndarray, float]:
    """Initialize a weight matrix with zeros

    Parameters:
      feature (int): Number of features
      output (int): Number of outputs (next layer neurons nbr)

    Returns:
        tuple(np.ndarray, float): Weights matrix and bias
    """
    return np.zeros((features, output)), 0.0


def he_initialisation(features: int, output: int, seed: int
                      ) -> tuple[np.ndarray, float]:
    """Initialize a weight matrix with He initialization

    Parameters:
      feature (int): Number of features
      output (int): Number of outputs (next layer neurons nbr)
      seed (int): Random seed

    Returns:
        tuple(np.ndarray, float): Weights matrix and bias
    """
    rng = np.random.default_rng(seed=seed)
    return (rng.standard_normal(size=(features, output)) *
            np.sqrt(2 / output), 0.0)


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


def score_function(weights: np.ndarray,
                   features: np.ndarray,
                   bias: np.ndarray | float | None = None
                   ) -> np.ndarray | float:
    """Calculate score function for one sample
    thetas and features has to be the same size

    Parameters:
      weights (np.ndarray): Weights values
      features (np.ndarray): Features values
      bias (np.ndarray) (optionnal): Used if specified instead of feature
                                     first col

    Returns:
      np.ndarray | float: Score value array | Score value
    """
    res = weights @ features if bias is None else weights @ features + bias
    return res


def categorical_cross_entropy(prediction: np.ndarray, truth: np.ndarray):
    """Calculate CrossEntropy for model tunning (loss)

    Parameters:
      prediction (np.ndarray): Values predicted by the model
      truth (np.ndarray): True value from the dataset

    Return:
      (np.ndarray): CrossEntropy result for each input
    """
    epsilon = 1e-9
    clipped_prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
    loged_prediction = np.log(clipped_prediction)
    loss = truth * loged_prediction
    return -np.sum(loss, axis=1)


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """Calculate sigmoid function for one sample

    Parameters:
      values (numpy.ndarray | float): Values to use for the sigmoid function

    Returns:
      np.ndarray | float: Sigmoid value
    """
    return 1 / (1 + np.exp(-values))


def softmax(z: np.ndarray) -> np.ndarray:
    """Compute the softmax of a vector.
    Parameters:
      z (numpy.ndarray): Input vector.

    Return:
      numpy.ndarray: Softmax of the input vector.
    """
    z_max = ftm.ft_max(z)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1)[:, None]


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


def perceptron(inputs: np.ndarray,
               weights: np.ndarray,
               bias: float,
               activation=sigmoid
               ) -> np.ndarray:
    """Single perceptron

    Parameters:
      inputs (np.ndarray): Input data (matrix)
      weights (np.ndarray): Weights of the perceptron
      bias (float): Bias of the perceptron
      activation (function): Activation function to use

    Returns:
      np.ndarray: Output of the perceptron
    """
    weighted_sum = np.dot(weights, inputs) + bias
    return activation(weighted_sum)


def hidden_layer(inputs: np.ndarray,
                 weights: np.ndarray,
                 bias: np.ndarray,
                 activation=sigmoid
                 ) -> np.ndarray:
    """Compute the output of a hidden layer

    Parameters:
    input (np.ndarray): Input data (matrix)
    weights (np.ndarray): Weights of the layer (matrix)
    bias (np.ndarray): Bias of the layer
    activation (function) (optionnal): Activation function to use (default:
                                       sigmoid)

    Returns:
      np.ndarray: Output of the layer (matrix)
    """
    weighted_sum = inputs @ weights + bias
    return activation(weighted_sum)
