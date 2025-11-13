# Multilayer Perceptron - TODO List

## Training
- [x] Make loss_result as an array to track values over epochs
- [x] Add learning curves visualization with matplotlib
  - [x] Display loss curves (training vs validation)
  - [x] Display accuracy curves (training vs validation)
- [x] Implement weight and model saving (JSON format)
- [x] Rework the train.py inputs to fit new model signature
- [x] Rework model template for better readability (losses and accuracy)
- [x] Pure feedoforward function ? would allow for batching
- [x] Add ability to change optimizer function (Verify and update)
  - [x] Batch gradient descent (batch = data size)
  - [x] Mini-batch gradient descent (1 < batch < data_size)
  - [x] Stochastic gradient descent (batch = 1)

## Predicting
- [x] Update model template to remove data_* fields
- [x] Implement model weight + layer dimension verification
- [x] Implement predict functionality
- [x] Create predict.py program

## Bug Fixes & Testing
- [x] Fix model optimizer field
- [ ] Test
- [x] Change parser file type to str instead of IO file

## Model Enhancement
- [x] Add features field when saving/loading models
- [x] Add feature selector from json
- [x] Rework velocity initialization for each layer
- [x] Create weight velocity + bias velocity initialization
- [ ] Add and check for mometum (beta) field in model when using optimizer like nesterov
- [ ] Implement nestrov optimisation using Ilya optimisation for velocity and weight updates

## Bonus Features
> **Note**: Bonus features will only be implemented after the mandatory part is PERFECT

### Advanced Optimization
- [ ] Implement advanced optimization functions beyond basic gradient descent
  - [ ] Nesterov momentum optimizer
  - [ ] RMSprop optimizer  
  - [ ] Adam optimizer

### Enhanced Visualization & Analysis
- [ ] Add display of multiple learning curves on the same graph for model comparison
- [ ] Implement training metrics history tracking and storage
- [ ] Evaluate learning phase with multiple metrics (beyond loss)
  - [ ] Accuracy tracking
  - [ ] Precision, Recall, F1-score
  - [ ] ROC-AUC metrics

### Training Improvements
- [ ] Add early stopping functionality to prevent overfitting
  - [ ] Monitor validation loss plateau
  - [ ] Configurable patience parameter
  - [ ] Best model checkpoint saving

