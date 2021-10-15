import numpy as np
from common.problem_type import problem_type
from reader import prepare_data

first_colors = ['darkred','darkgreen','darkblue']
second_colors = ['indianred','forestgreen','royalblue']

def generate_classification_graph(dataset):
    d = np.array([data[0] for data in dataset])
    x_max, y_max = np.max(d, axis=0)
    x_min, y_min = np.min(d, axis=0)

def generate_regression_graph(path_to_train_dataset, path_to_test_dataset):
    pass


if __name__ == "__main__":
    dataset = prepare_data(problem_type.Classification, "data/classification/data.simple.test.100.csv")
    generate_classification_graph(dataset)