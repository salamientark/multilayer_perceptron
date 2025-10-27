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

## Model Enhancement
- [ ] Add features field when saving/loading models

