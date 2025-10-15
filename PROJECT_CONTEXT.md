### - - - PROJECT CONTEXT - - -

**Objective**: Implement a multilayer perceptron neural network from scratch to classify breast cancer as malignant or benign using the Wisconsin dataset.

**Key Features**:
- Feedforward neural network with minimum 2 hidden layers
- Backpropagation and gradient descent training algorithms
- Softmax output layer for binary classification
- Data preprocessing and train/validation split
- Learning curve visualization (loss/accuracy graphs)
- Model persistence (save/load trained weights)

**Core Components**:
- Dataset splitter program
- Training program with epoch-based learning
- Prediction program with performance evaluation
- Binary cross-entropy loss function implementation

**Technical Constraints**:
- Must be implemented from scratch (no ML/neural network libraries)
- Can use linear algebra libraries (numpy) and plotting libraries
- Language choice is free (Python recommended based on existing files)
- Must handle 32-column Wisconsin breast cancer dataset
- Required output format: epoch-by-epoch training metrics

**Data Specifications**:
- Input: 30 numerical features describing cell nucleus characteristics  
- Output: Binary classification (Malignant/Benign)
- Dataset format: CSV with diagnosis column as target label
- Preprocessing required for raw data

**Expected Deliverables**:
- Modular codebase with clear separation of concerns
- Training visualization with loss/accuracy curves
- Saved model file (weights and topology)
- Performance evaluation using binary cross-entropy
- Clear understanding of feedforward, backpropagation, gradient descent concepts
