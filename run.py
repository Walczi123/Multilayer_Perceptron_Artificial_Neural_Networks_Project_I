from common.functions import function_type
from perceptron import MLP
from common.problem_type import problem_type
from common.reader import normalize, prepare_data
from common.graphs import generate_classification_graph_for_model, generate_loss_function_graph, generate_regression_graph, generate_classification_graph_of_points
import numpy as np
from matplotlib import pyplot

# Parameters

PROBLEM_TYPE = problem_type.Regression
PATH_TO_TRAIN_DATASET = "data/regression/data.cube.train.100.csv"
PATH_TO_TEST_DATASET = "data/regression/data.cube.test.100.csv"
# PATH_TO_TRAIN_DATASET = "data/classification/data.simple.train.100.csv"
# PATH_TO_TEST_DATASET = "data/classification/data.simple.test.100.csv"
# PATH_TO_TRAIN_DATASET = "data/classification/data.three_gauss.train.100.csv"
# PATH_TO_TEST_DATASET = "data/classification/data.three_gauss.test.100.csv"


LAYERS = [1, 32, 16, 1]
ACTIVATION_FUNCTION = function_type.Sigmoid
# OUTPUT_FUNCTION = function_type.Softmax
OUTPUT_FUNCTION = function_type.Simple
EPOCHS = 100
ITERATIONS = 100
LEARINN_RATE = 0.01
SEED = 141
SHOW_PERCENTAGE = 1
BIAS = True

if __name__ == "__main__":
    perceptron = MLP(PROBLEM_TYPE, LAYERS, ACTIVATION_FUNCTION, OUTPUT_FUNCTION,
                     EPOCHS, LEARINN_RATE, SEED, BIAS)
    train_dataset = prepare_data(PROBLEM_TYPE, PATH_TO_TRAIN_DATASET)
    test_dataset = prepare_data(PROBLEM_TYPE, PATH_TO_TEST_DATASET)

    if PROBLEM_TYPE == problem_type.Regression:
        train_dataset, test_dataset = normalize(train_dataset, test_dataset)

    perceptron.train(train_dataset, SHOW_PERCENTAGE)
    _, _, predictions = perceptron.test(test_dataset, SHOW_PERCENTAGE)

    if PROBLEM_TYPE == problem_type.Regression:
        x = []
        y = []
        for i in range(len(test_dataset)):
            x.append(test_dataset[i][0])
            y.append(test_dataset[i][1])
        tx = []
        ty = []
        for i in range(len(train_dataset)):
            tx.append(train_dataset[i][0])
            ty.append(train_dataset[i][1])
        generate_regression_graph((x, y), (x, predictions), (tx, ty))
    elif PROBLEM_TYPE == problem_type.Classification:
        points = []
        train_points = []
        for i in range(len(test_dataset)):
            points.append((test_dataset[i][0], predictions[i]))
        for i in range(len(train_dataset)):
            train_points.append(
                (train_dataset[i][0], np.nonzero(train_dataset[i][1])))

        # generate_classification_graph_of_points(points, train_points)
        generate_classification_graph_for_model(perceptron, train_dataset, test_dataset)


    # LOSS FUNCTION
    # perceptron.epochs = 1
    # epochs=[]
    # train_loss= []
    # test_loss= []
    # step = 2
    # for i in range(ITERATIONS):
    #     perceptron.train(train_dataset, -1)
    #     if i%step or i == 0:
    #         print(i)
    #         epochs.append(i)
    #         test_loss.append(perceptron.test(test_dataset, -1)[1])
    #         train_loss.append(perceptron.test(train_dataset, -1)[1]+10)
        
    # generate_loss_function_graph(epochs, train_loss, test_loss)
