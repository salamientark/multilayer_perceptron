import pandas as pd
from .create_model import load_model_from_json
from .preprocessing import standardize_df, get_class_list, one_encode
from .model_utils import load_weights_from_file


def load_predict_model_weights(model: dict, weights):
    """Load weights into model structure

    Used for prediction.

    Parameters:
      model (dict) : Model structure
      weights (np.lib.npyio.NpzFile) : Weights loaded from file
    """
    for i, layer in enumerate(model['layers']):
        weight_key = f'layer_{i}_weights'
        bias_key = f'layer_{i}_bias'
        layer['weights'] = weights[weight_key]
        layer['bias'] = weights[bias_key]
    model['output']['weights'] = weights['output_weights']
    model['output']['bias'] = weights['output_bias']


def load_predict_model_data(
        model: dict,
        dataset,
        features: list = [],
        target: str | None = None
        ) -> None:
    """Load dataset into model structure

    Used for prediction.

    Parameters:
      model (dict) : Model structure
      dataset (pandas.DataFrame) : Dataset to load
      features (list) : List of features names
    target (str | None) : Target column name

    Returns:
      dict : Model structure with loaded dataset
    """
    df = pd.read_csv(dataset)
    if not features:
        features = df.columns.tolist()
    filtered_df = pd.DataFrame(df[features])
    standardized_data = standardize_df(filtered_df)
    model['data'] = standardized_data.to_numpy()
    if target is not None:
        target_classes = get_class_list(df, target)
        model['truth_classes'] = target_classes
        model['raw_truth'] = df[target].to_list()
        model['truth'] = one_encode(df, target)


def load_predict_model(
        model_filename: str,
        weights_filename: str,
        data_filename: str,
        features: list = [],
        target: str | None = None
        ) -> dict:
    """Load model, weights, and dataset for prediction

    Parameters:
      model_filename (str) : Path to model file
      weights_filename (str) : Path to weights file
      data_filename (str) : Path to dataset file
      features (list) : List of feature names
      target (str | None) : Target column name (not used here)

    Returns:
      dict : Loaded model structure with weights and dataset
    """
    model = load_model_from_json(model_filename)
    weights = load_weights_from_file(weights_filename)
    load_predict_model_weights(model, weights)

    # Load dataset
    if model['features'] is not None:
        features = model['features']
    if model['target'] is not None:
        target = model['target']
    load_predict_model_data(model, data_filename, features, target)

    # Remove unneeded keys
    for layer in model['layers']:
        layer.pop('weights_initializer', None)
    model['output'].pop('weights_initializer', None)
    return model
