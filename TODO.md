# Multilayer Perceptron - TODO List

## Training
- [x] Make loss_result as an array to track values over epochs
- [x] Add learning curves visualization with matplotlib
  - [x] Display loss curves (training vs validation)
  - [x] Display accuracy curves (training vs validation)
- [x] Implement weight and model saving (JSON format)
- [x] Rework the train.py inputs to fit new model signature
- [x] Rework model template for better readability (losses and accuracy)
- [ ] Pure feedoforward function ? would allow for batching
- [ ] Add ability to change optimizer function
  - [ ] Batch gradient descent
  - [ ] Mini-batch gradient descent
  - [ ] Stochastic gradient descent

## Predicting
- [ ] Create predict.py program
