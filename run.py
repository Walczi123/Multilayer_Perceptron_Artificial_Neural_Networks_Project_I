from common.functions import function_type
from perceptron import MLP
from common.problem_type import problem_type
from reader import prepare_data
import numpy as np

# Parameters

PROBLEM_TYPE = problem_type.Regression
PATH_TO_TRAIN_DATASET = "data/regression/data.cube.train.100.csv"
PATH_TO_TEST_DATASET = "data/regression/data.cube.test.100.csv"
# PATH_TO_TRAIN_DATASET = "data/classification/data.simple.test.100.csv"
# PATH_TO_TEST_DATASET = "data/classification/data.simple.test.100.csv"


LAYERS = [1, 3, 2]
ACTIVATION_FUNCTION = function_type.Sigmoid
TRANSFER_FUNCTION = function_type.Sigmoid
EPOCHS = 10
LEARINN_RATE = 0.3
LEARINN_COEFFICIENT = 0.3
SEED = 141
SHOW_PERCENTAGE = 101
BIAS = False

# D = []
# for i in range(0, 199):
#     D.append([np.array(i), TRANSFER_FUNCTION(i)])


perceptron = MLP(LAYERS, ACTIVATION_FUNCTION, TRANSFER_FUNCTION,
                 EPOCHS, LEARINN_RATE, LEARINN_COEFFICIENT, SEED, BIAS)
perceptron.train(prepare_data(
    PROBLEM_TYPE, PATH_TO_TRAIN_DATASET), SHOW_PERCENTAGE)
perceptron.test(prepare_data(
    PROBLEM_TYPE, PATH_TO_TEST_DATASET), SHOW_PERCENTAGE)

# perceptron.train(D, SHOW_PERCENTAGE)
# perceptron.test(D, SHOW_PERCENTAGE)
