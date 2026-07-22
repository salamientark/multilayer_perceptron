# Multilayer Perceptron

A from-scratch implementation of a multilayer perceptron neural network for binary classification on the Wisconsin breast cancer dataset.

Goal of the project is described in multilayer_perceptron.pdf

## Project Overview

This project implements a multilayer perceptron (MLP) neural network from scratch using only numpy for linear algebra operations. The network is trained to classify breast tumors as malignant (M) or benign (B) based on 30 features describing cell nucleus characteristics.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package & environment manager
- Python **3.10+** (uv installs it automatically if missing)

Python dependencies (installed by `uv sync`):

| Package    | Purpose                          |
|------------|----------------------------------|
| numpy      | Linear algebra / matrix ops      |
| pandas     | CSV loading / data handling      |
| matplotlib | Learning curves and plots        |
| seaborn    | Pairplots and correlation heatmap|
| PyQt5      | Matplotlib interactive backend   |

## How to use

### Setup
First you must create the python environment locally, install dependencies and activate this virtual environment for the current session.
```bash
uv sync
source .venv/bin/activate
```

`uv sync` creates the `.venv` virtual environment and installs all
dependencies from `pyproject.toml` / `uv.lock`. It also installs the `ft_mlp`
package itself in editable mode, so the programs are importable from anywhere.
`make` is kept as an alias for `uv sync`.

### Running the programs

Each program can be run in three equivalent ways:

```bash
# 1. As a module (works from any directory)
uv run -m ft_mlp.train --help

# 2. As a script path
uv run src/ft_mlp/train.py --help

# 3. As plain python, with the venv activated
python src/ft_mlp/train.py --help
```

Installed console scripts are also available: `ft-train`, `ft-predict`,
`ft-split-dataset`, `ft-analyse-data`.

The examples below use the `uv run -m` form. Substitute any of the above.

### Basic Workflow

1. **Analyze the dataset** (optional):
```bash
uv run -m ft_mlp.analyse_data data.csv
```

2. **Split the dataset**:
```bash
uv run -m ft_mlp.split_dataset data.csv
```

3. **Train the model**:
```bash
uv run -m ft_mlp.train --shape 24 16 --epoch 100 --learning_rate 0.1 --seed 42 data_training.csv
```

4. **Make predictions**:
```bash
uv run -m ft_mlp.predict --model trained_model.json --weights weights.npz --data data_validation.csv
```

#### How the data is split

There are two levels of splitting, which is deliberate:

- `split_dataset` holds out `data_validation.csv`. That file is **never seen
  during training** — not by the weights, and not by the standardization
  statistics. It is what `predict` is evaluated on.
- `train` splits the file it is given once more (`--train_ratio`, default 0.8)
  into an inner training set and an inner validation set. The inner validation
  set is what produces the per-epoch `val_loss` and the learning curves, and it
  is what any hyperparameter choice is based on.

So of the 569 samples: 455 go to `train`, which uses 364 to fit the weights and
91 to watch for overfitting, and the remaining 114 stay untouched for `predict`.
Standardization statistics are fitted on the inner training set alone and stored
in `trained_model.json`, so `predict` rescales new data with the training
statistics rather than recomputing them from whatever file it is handed.

## Project structure

```
.
├── src/ft_mlp/
│   ├── split_dataset.py   # Split data into train/validation sets
│   ├── analyse_data.py    # Dataset statistics and plots
│   ├── train.py           # Train the network (backpropagation)
│   ├── predict.py         # Predict with a trained model
│   ├── network_layers.py  # Layers, activations, forward pass
│   ├── loss_functions.py  # Cross-entropy losses
│   ├── preprocessing.py   # Standardization, encoding, splitting
│   ├── create_model.py    # Model construction from args/JSON
│   ├── model_utils.py     # Save/load weights, metrics
│   ├── initializer.py     # Weight initialization
│   ├── analysis.py        # Describe / correlation matrix
│   └── ft_math.py         # From-scratch math primitives
├── tests/                 # Unit tests
├── scripts/run_tests.py   # Test runner (flake8 + unittest)
├── pyproject.toml         # Project metadata & dependencies
└── uv.lock                # Pinned dependency versions
```

## Programs Description

### 1. `src/ft_mlp/split_dataset.py`

Splits the dataset into training and validation sets.

**Usage:**
```bash
uv run -m ft_mlp.split_dataset <dataset.csv> [OPTIONS]
```

**Arguments:**
- `dataset_path` (required): Path to the input CSV file
  - Pass it *before* `--outfile`: `--outfile` accepts several values, so it
    swallows the dataset path if it comes first.

**Options:**
- `--outfile`, `-o`: Output filenames (default: data_training.csv data_validation.csv)
  - Can be comma-separated string or two separate arguments
- `--seed`, `-s`: Random seed for shuffling (default: 1)
- `--train-ratio`, `-r`: Ratio of training set size (default: 0.8, range: 0.0-1.0)

**Example:**
```bash
uv run -m ft_mlp.split_dataset --seed 42 --train-ratio 0.8 data.csv
```

### 2. `src/ft_mlp/analyse_data.py`

Analyzes and visualizes the dataset with statistics and correlation plots.

**Usage:**
```bash
uv run -m ft_mlp.analyse_data <dataset.csv>
```

**Features:**
- Displays descriptive statistics (mean, std, min, max, quartiles)
- Generates pairplots for mean, std, and worst features
- Creates correlation heatmap showing feature relationships

### 3. `src/ft_mlp/train.py`

Trains the multilayer perceptron model.

**Usage:**
```bash
uv run -m ft_mlp.train [OPTIONS] <dataset>
```

**Required Arguments (mutually exclusive):**
- `--shape`: List of integers defining neurons per hidden layer
  - Example: `--shape 24 16` creates 2 hidden layers with 24 and 16 neurons
- `--layer`: Accepts either form
  - A list of widths: `--layer 24 24 24` creates 3 hidden layers of 24 neurons
    (this is the syntax used in the subject; equivalent to `--shape 24 24 24`)
  - A single count, combined with `--neurons`: `--layer 3 --neurons 20` creates
    3 hidden layers with 20 neurons each
- `--conf`: Path to model configuration JSON file

**Required Options:**
- `--epoch`, `--epochs`, `-e`: Number of training iterations (must be > 0)
- `--learning_rate`, `-a`: Learning rate (range: 0.0-1.0)
- `--seed`, `-s`: Random seed for reproducibility (must be positive integer)

**Optional Arguments:**
- `--features`: Subset of features to use (default: all 30 features)
- `--loss`: Loss function (choices: 'categoricalCrossentropy', default: 'categoricalCrossentropy')
- `--batch`, `--batch_size`, `-b`: Batch size for mini-batch gradient descent (default: full batch)
  - If batch=1: stochastic gradient descent
  - If 1 < batch < dataset_size: mini-batch gradient descent
  - If batch >= dataset_size: batch gradient descent
- `--train_ratio`, `-tr`: Training/validation split ratio (default: 0.8)
- `--outfile`, `-of`: Output filename for weights (default: "weights.npz")
- `--model_outfile`, `-mo`: Output filename for the model topology (default: "trained_model.json")
- `--plot_outfile`, `-po`: Save the learning curves to an image file instead of
  opening a window. Use this on a machine without a display.

**Examples:**
```bash
# Basic training with 2 hidden layers
uv run -m ft_mlp.train --shape 24 16 --epoch 100 --learning_rate 0.1 --seed 42 data_training.csv

# Training with mini-batch gradient descent
uv run -m ft_mlp.train --shape 24 16 --epoch 100 -a 0.1 -s 42 --batch 32 data_training.csv

# Training with uniform hidden layers
uv run -m ft_mlp.train --layer 3 --neurons 20 --epoch 100 -a 0.1 -s 42 data_training.csv

# The subject's example CLI (IV.3), accepted verbatim
uv run -m ft_mlp.train data_training.csv --layer 24 24 24 --epochs 84 \
    --loss categoricalCrossentropy --batch_size 8 --learning_rate 0.0314
```

> **Argument order:** `--shape` and `--layer` take a variable number of values,
> so put the dataset *before* them (or separate them from it with another
> option). `--layer 24 24 24 data_training.csv` makes argparse try to read the
> filename as a layer width and it exits with
> `argument --layer: invalid int value: 'data_training.csv'`.

### 4. `src/ft_mlp/predict.py`

Makes predictions using a trained model.

**Usage:**
```bash
uv run -m ft_mlp.predict --model <model.json> --weights <weights.npz> --data <dataset.csv>
```

**Required Arguments:**
- `--model`, `-m`: Path to model structure JSON file
- `--weights`, `-w`: Path to model weights NPZ file
- `--data`, `-d`: Path to dataset for prediction

**Output:**
- Creates `prediction.csv` with predictions for each sample
- Displays prediction accuracy and binary cross-entropy loss

**Example:**
```bash
uv run -m ft_mlp.predict -m trained_model.json -w weights.npz -d data_validation.csv
```

## Core Functions Reference

### Network Architecture Functions

#### `hidden_layer(inputs, weights, bias, activation=sigmoid)`
Computes the output of a neural network layer.

**Parameters:**
- `inputs` (np.ndarray): Input data matrix of shape (batch_size, input_features)
- `weights` (np.ndarray): Weight matrix of shape (input_features, neurons)
- `bias` (np.ndarray): Bias vector of shape (neurons,)
- `activation` (function): Activation function (default: sigmoid)

**Returns:**
- `np.ndarray`: Activated output of shape (batch_size, neurons)

**Location:** src/ft_mlp/network_layers.py:87

---

#### `sigmoid(values)`
Applies sigmoid activation function element-wise.

**Parameters:**
- `values` (np.ndarray | float): Input values

**Returns:**
- `np.ndarray | float`: Sigmoid output in range (0, 1)

**Formula:** σ(x) = 1 / (1 + e^(-x))

**Location:** src/ft_mlp/network_layers.py:25

---

#### `softmax(z)`
Computes softmax activation for multi-class classification output layer.

**Parameters:**
- `z` (np.ndarray): Input matrix of shape (batch_size, num_classes)

**Returns:**
- `np.ndarray`: Probability distribution over classes (sums to 1 per sample)

**Formula:** softmax(x_i) = e^(x_i - max(x)) / Σ(e^(x_j - max(x)))

**Location:** src/ft_mlp/network_layers.py:54

---

#### `predict(model, inputs)`
Makes predictions using the trained model (forward pass only).

**Parameters:**
- `model` (dict): Model structure with layers, weights, and biases
- `inputs` (np.ndarray): Input features of shape (batch_size, features)

**Returns:**
- `np.ndarray`: Predictions of shape (batch_size, num_classes)

**Note:** Use this for inference only, not during training.

**Location:** src/ft_mlp/network_layers.py:108

---

### Training Functions

#### `feed_forward(model, inputs)`
Performs forward pass and stores all layer activations for backpropagation.

**Parameters:**
- `model` (dict): Model structure
- `inputs` (np.ndarray): Input batch

**Returns:**
- `list[np.ndarray]`: Activation outputs from each layer (including output layer)

**Location:** src/ft_mlp/train.py:207

---

#### `backpropagation(model, inputs, results, truth)`
Computes gradients for all layers using backpropagation algorithm.

**Parameters:**
- `model` (dict): Model structure with weights and activation functions
- `inputs` (np.ndarray): Input data used in forward pass
- `results` (list[np.ndarray]): Layer activations from feed_forward
- `truth` (np.ndarray): One-hot encoded ground truth labels

**Returns:**
- `list[tuple[np.ndarray, np.ndarray]]`: List of (weight_gradients, bias_gradients) for each layer

**Algorithm:**
1. Compute output layer gradient: ∂L/∂z = predictions - truth
2. Propagate gradient backwards through each layer
3. Apply chain rule with activation derivatives

**Location:** src/ft_mlp/train.py:235

---

#### `update_weights(model, gradients)`
Updates model weights using gradient descent.

**Parameters:**
- `model` (dict): Model to update (modified in-place)
- `gradients` (list[tuple]): Weight and bias gradients from backpropagation

**Formula:** 
- w_new = w_old - α * ∂L/∂w
- b_new = b_old - α * ∂L/∂b

**Location:** src/ft_mlp/train.py:296

---

### Loss Functions

#### `categorical_cross_entropy(prediction, truth)`
Computes categorical cross-entropy loss for multi-class classification.

**Parameters:**
- `prediction` (np.ndarray): Predicted probabilities (batch_size, num_classes)
- `truth` (np.ndarray): One-hot encoded labels (batch_size, num_classes)

**Returns:**
- `np.ndarray`: Loss value for each sample in batch

**Formula:** L = -Σ(y_true * log(y_pred))

**Location:** src/ft_mlp/loss_functions.py:4

---

#### `binary_cross_entropy(prediction, truth)`
Computes binary cross-entropy loss (used for evaluation).

**Parameters:**
- `prediction` (np.ndarray): Predicted probabilities
- `truth` (np.ndarray): Ground truth labels

**Returns:**
- `np.ndarray`: BCE loss for each sample

**Formula:** L = -(y*log(p) + (1-y)*log(1-p))

**Location:** src/ft_mlp/loss_functions.py:21

---

### Preprocessing Functions

#### `standardize_df(df, columns=[])`
Standardizes dataframe columns using z-score normalization.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `columns` (list): Columns to standardize (default: all numerical columns)

**Returns:**
- `pd.DataFrame`: Standardized dataframe

**Formula:** z = (x - μ) / σ

**Location:** src/ft_mlp/preprocessing.py:196

---

#### `split_dataset(df, ratio=0.8, seed=1)`
Splits dataframe into training and validation sets.

**Parameters:**
- `df` (pd.DataFrame): Dataset to split
- `ratio` (float): Training set ratio (default: 0.8)
- `seed` (int): Random seed for reproducibility

**Returns:**
- `tuple[pd.DataFrame, pd.DataFrame]`: (training_set, validation_set)

**Location:** src/ft_mlp/preprocessing.py:220

---

#### `one_encode(df, col)`
Converts categorical labels to one-hot encoded format.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `col` (str): Column name containing class labels

**Returns:**
- `np.ndarray`: One-hot encoded array of shape (samples, num_classes)

**Example:** ['M', 'B', 'M'] → [[1, 0], [0, 1], [1, 0]]

**Location:** src/ft_mlp/preprocessing.py:83

---

### Model Management Functions

#### `create_model(args, target, features=None)`
Creates and initializes the complete model structure.

**Parameters:**
- `args` (argparse.Namespace): Parsed command-line arguments
- `target` (str): Name of target column in dataset
- `features` (list): List of feature names to use (default: all features)

**Returns:**
- `dict`: Complete model structure with:
  - Network architecture (layers, neurons)
  - Training parameters (learning rate, batch size, epochs)
  - Datasets (training and validation)
  - Optimizer configuration

**Location:** src/ft_mlp/create_model.py:208

---

#### `init_model(model)`
Initializes model weights and biases using He initialization.

**Parameters:**
- `model` (dict): Model structure to initialize

**Returns:**
- `dict`: Model with initialized weights and bias arrays

**Weight Initialization:** He Normal (w ~ N(0, sqrt(2/n_inputs)))

**Location:** src/ft_mlp/train.py:171

---

#### `save_model(filename, model)`
Saves model structure to JSON file.

**Parameters:**
- `filename` (str): Output filename (e.g., "model.json")
- `model` (dict): Model structure to save

**Location:** src/ft_mlp/model_utils.py:164

---

#### `save_weights(filename, model)`
Saves model weights and biases to compressed numpy file.

**Parameters:**
- `filename` (str): Output filename (e.g., "weights.npz")
- `model` (dict): Model with trained weights

**Location:** src/ft_mlp/model_utils.py:129

---

#### `load_predict_model(model_file, weights_file, data_file, features, target)`
Loads trained model for making predictions.

**Parameters:**
- `model_file` (str): Path to model JSON file
- `weights_file` (str): Path to weights NPZ file
- `data_file` (str): Path to prediction dataset
- `features` (list): Feature column names
- `target` (str): Target column name

**Returns:**
- `dict`: Loaded model ready for prediction

**Location:** src/ft_mlp/load_predict_model.py

---

### Weight Initialization

#### `he_initialisation(features, output, seed, inputs)`
Initializes weights using He initialization (optimal for ReLU/sigmoid).

**Parameters:**
- `features` (int): Number of input features
- `output` (int): Number of output neurons
- `seed` (int): Random seed
- `inputs` (int): Number of training samples (used for scale calculation)

**Returns:**
- `tuple[np.ndarray, np.ndarray]`: (initialized_weights, zero_bias)

**Formula:** w ~ N(0, sqrt(2/n_inputs))

**Location:** src/ft_mlp/initializer.py:17

---

### Utility Functions

#### `calculate_accuracy(predictions, truth)`
Computes classification accuracy.

**Parameters:**
- `predictions` (np.ndarray): Model predictions (probabilities)
- `truth` (np.ndarray): One-hot encoded ground truth

**Returns:**
- `float`: Accuracy as ratio of correct predictions (0.0-1.0)

**Location:** src/ft_mlp/model_utils.py:210

---

#### `get_random_batch_indexes(data_size, seed=None)`
Generates shuffled indices for mini-batch training.

**Parameters:**
- `data_size` (int): Total number of samples
- `seed` (int): Random seed (optional)

**Returns:**
- `np.ndarray`: Permuted indices array

**Location:** src/ft_mlp/model_utils.py:49

---

## Model Architecture

The neural network consists of:

1. **Input Layer**: 30 neurons (one per feature)
2. **Hidden Layers**: User-defined (recommended: 2+ layers)
   - Activation: Sigmoid
   - Initialization: He Normal
3. **Output Layer**: 2 neurons (binary classification)
   - Activation: Softmax
   - Outputs probability distribution over classes

## Training Process

1. **Initialization**: Weights initialized using He initialization
2. **Forward Pass**: Compute activations layer by layer
3. **Loss Calculation**: Categorical cross-entropy loss
4. **Backward Pass**: Backpropagation to compute gradients
5. **Weight Update**: Gradient descent with learning rate α
6. **Validation**: Evaluate on held-out validation set

## Dataset Information

**Wisconsin Breast Cancer Dataset:**
- 569 samples total
- 30 features describing cell nucleus characteristics
- Binary classification: M (Malignant) or B (Benign)
- Features include radius, texture, perimeter, area, smoothness, compactness, concavity, etc.
- Three sets of 10 features each: mean, standard error, and worst values

## Implementation Details

**Allowed Libraries:**
- numpy (linear algebra)
- pandas (data manipulation)
- matplotlib/seaborn (visualization)

**From-Scratch Components:**
- Neural network architecture
- Forward propagation
- Backpropagation algorithm
- Gradient descent optimization
- Activation functions (sigmoid, softmax)
- Loss functions (categorical cross-entropy)
- Weight initialization (He)


