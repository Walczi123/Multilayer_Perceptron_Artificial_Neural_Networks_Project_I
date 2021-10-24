from common.functions import function_type
from perceptron import MLP
from common.problem_type import problem_type
from common.reader import prepare_data
from common.graphs import generate_regression_graph, generate_classification_graph_of_points
import numpy as np

# Parameters

PROBLEM_TYPE = problem_type.Regression
PATH_TO_TRAIN_DATASET = "data/regression/data.cube.train.100.csv"
PATH_TO_TEST_DATASET = "data/regression/data.cube.test.100.csv"
# PATH_TO_TRAIN_DATASET = "data/classification/data.simple.train.100.csv"
# PATH_TO_TEST_DATASET = "data/classification/data.simple.test.100.csv"


LAYERS = [1, 3, 1]
ACTIVATION_FUNCTION = function_type.Tanh
TRANSFER_FUNCTION = function_type.Simple
EPOCHS = 1000
LEARINN_RATE = 0.1
LEARINN_COEFFICIENT = 0.1
SEED = 141
SHOW_PERCENTAGE = 1
BIAS = True

if __name__ == "__main__":
    perceptron = MLP(LAYERS, ACTIVATION_FUNCTION, TRANSFER_FUNCTION,
                     EPOCHS, LEARINN_RATE, LEARINN_COEFFICIENT, SEED, BIAS)
    train_dataset = prepare_data(PROBLEM_TYPE, PATH_TO_TRAIN_DATASET)
    test_dataset = prepare_data(PROBLEM_TYPE, PATH_TO_TEST_DATASET)

    # train_dataset = []
    # for i in range(0, 199):
    #     train_dataset.append([np.array([i/20]), TRANSFER_FUNCTION(i/20)])

    # test_dataset = []
    # for i in range(0, 199):
    #     test_dataset.append([np.array([i/20]), TRANSFER_FUNCTION(i/20)])

    # train_dataset = []
    # for i in range(0, 199):
    #     train_dataset.append([np.array([i]), i*i])

    # test_dataset = []
    # for i in range(0, 199):
    #     test_dataset.append([np.array([i]), i*i])

    # test_dataset = train_dataset

    perceptron.train(train_dataset, SHOW_PERCENTAGE)
    _, _, predictions = perceptron.test(test_dataset, SHOW_PERCENTAGE)

    x = []
    y = []
    for i in range(len(test_dataset)):
        x.append(test_dataset[i][0])
        y.append(test_dataset[i][1])

    xt = []
    yt = []
    for i in range(len(train_dataset)):
        xt.append(test_dataset[i][0])
        yt.append(test_dataset[i][1])

    generate_regression_graph((x, y), (x, predictions), (xt, yt))
    # generate_classification_graph_of_points(predictions)
