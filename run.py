from common.functions import function_type
from perceptron import MLP
from common.problem_type import problem_type
from reader import prepare_data

# Parameters

PROBLEM_TYPE = problem_type.Regression
PATH_TO_TRAIN_DATASET = "data/regression/data.activation.test.100.csv"
PATH_TO_TEST_DATASET = "data/regression/data.activation.test.100.csv"


LAYERS = [1, 1, 1]
ACTIVATION_FUNCTION = function_type.Simple
TRANSFER_FUNCTION = function_type.Simple
EPOCHS = 2
LEARINN_RATE = 0.5
LEARINN_COEFFICIENT = 0.5
SEED = 0
SHOW_PERCENTAGE = 10
BIAS = True

perceptron = MLP(LAYERS, ACTIVATION_FUNCTION, TRANSFER_FUNCTION, EPOCHS, LEARINN_RATE, LEARINN_COEFFICIENT, SEED, BIAS)
perceptron.train(prepare_data(PROBLEM_TYPE, PATH_TO_TRAIN_DATASET), SHOW_PERCENTAGE)
perceptron.test(prepare_data(PROBLEM_TYPE, PATH_TO_TEST_DATASET), SHOW_PERCENTAGE)
