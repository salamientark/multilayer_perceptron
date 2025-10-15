# Agent Guidelines - Multilayer Perceptron

## Build/Test/Lint Commands
- `python train.py --help` - Display training options and parameters
- `python split_dataset.py` - Split dataset into training/validation sets
- `python train.py --dataset data_training.csv` - Train the neural network
- `python predict.py --model saved_model.npy --dataset data_test.csv` - Make predictions
- No specific linting requirements (norm not applied per project specs)

## Code Style Guidelines
- **Language**: Pure Python 3 (no restrictions on language choice)
- **Architecture**: Must implement from scratch - NO neural network libraries allowed
- **Allowed Libraries**: Linear algebra libraries (numpy), plotting libraries (matplotlib), data manipulation (pandas)
- **Modularity**: Code should be modular and well-structured
- **File Organization**: Free to organize files as desired within project constraints

## Neural Network Requirements
- **Minimum Architecture**: At least 2 hidden layers by default
- **Output Layer**: Must implement softmax function for probabilistic output
- **Activation Functions**: Implement sigmoid, tanh, ReLU or other activation functions
- **Training**: Must implement backpropagation and gradient descent from scratch
- **Loss Function**: Binary cross-entropy for evaluation

## Data Handling
- **Dataset**: Wisconsin breast cancer dataset (32 columns, M/B diagnosis)
- **Preprocessing**: Raw data must be preprocessed before training
- **Split**: Must separate into training and validation sets
- **Features**: 30 feature columns describing cell nucleus characteristics

## Implementation Structure
- **Dataset Splitter**: Program to divide data into train/validation
- **Training Program**: Implements backpropagation, saves model weights
- **Prediction Program**: Loads saved model, makes predictions, evaluates performance
- **Alternative**: Single program with mode switching options

## Output Requirements
- **Training Metrics**: Display loss and validation loss per epoch
- **Learning Curves**: Generate and display loss/accuracy graphs
- **Model Persistence**: Save network topology and learned weights
- **Evaluation**: Use binary cross-entropy error function for final evaluation

## Key Concepts to Understand
- **Feedforward**: Data flow from input to output layers
- **Backpropagation**: Error propagation for weight updates
- **Gradient Descent**: Optimization algorithm for learning
- **Bias**: Special neurons for controlling layer behavior
- **Weighted Sum**: Core perceptron computation with bias

## Error Handling
- Handle file I/O for dataset loading and model saving
- Validate input parameters and data shapes
- Provide clear error messages for invalid configurations
- Ensure reproducible results with optional seed parameter

## Performance Considerations
- Efficient matrix operations for forward/backward passes
- Memory management for large datasets
- Convergence monitoring and early stopping (bonus)
- Multiple optimization algorithms (bonus: Adam, RMSprop, etc.)
