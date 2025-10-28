# Multilayer perceptron

Goal of the project is described in multilayer_perceptron.pdf

## How I proceeded

#### Analyzing dataset

- First quick look of the data using dbeaver
The dataset is known and widely used for ML so we can easily find 
columns names.
- Describe database to see (mean, repartition, min/max...)
- Standardize data
- Plot pairplot to see correlation + correlation heatmap to see feature 
redundancy and predictive power

#### Splitting dataset

- We devide the dataset into two parts (train_dataset, test_dataset).
We'll use a optionnal argument that will be the seed of the dataset shuffle.
Those will allow to reproduce same splitting.

## How to use
First you must create the python environment locally, install dependencies and activate this virtual enviroment for the curret session.
```bash
make
source .venv/bin/activate
```


