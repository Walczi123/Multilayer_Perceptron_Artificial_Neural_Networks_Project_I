from common.functions import function_type
from perceptron import MLP
from common.problem_type import problem_type
from common.reader import normalize, prepare_data
from common.graphs import generate_classification_graph_for_model, generate_loss_function_graph, generate_regression_graph, generate_classification_graph_of_points
import numpy as np

# region Parameters

# region Datasets

PROBLEM_TYPE = problem_type.Regression
PATH_TO_TRAIN_DATASET = "data/regression/data.activation.train.1000.csv"
PATH_TO_TEST_DATASET = "data/regression/data.activation.test.1000.csv"
# PATH_TO_TRAIN_DATASET = "data/regression/data.cube.train.1000.csv"
# PATH_TO_TEST_DATASET = "data/regression/data.cube.test.1000.csv"

# PROBLEM_TYPE = problem_type.Classification
# PATH_TO_TRAIN_DATASET = "data/classification/data.simple.train.100.csv"
# PATH_TO_TEST_DATASET = "data/classification/data.simple.test.100.csv"
# PATH_TO_TRAIN_DATASET = "data/classification/data.three_gauss.train.100.csv"
# PATH_TO_TEST_DATASET = "data/classification/data.three_gauss.test.100.csv"

# endregion 

# region MPL Parameters

# Regression
ACTIVATION_FUNCTION = function_type.Sigmoid
OUTPUT_FUNCTION = function_type.Indentity
# LOSS_FUNCTION = function_type.MSE
LOSS_FUNCTION = function_type.MSLE

# Classification
# ACTIVATION_FUNCTION = function_type.Sigmoid
# OUTPUT_FUNCTION = function_type.Softmax
# LOSS_FUNCTION = function_type.Cross_entropy
# LOSS_FUNCTION = function_type.Hinge

LAYERS = [1, 32, 16, 1]
# LAYERS = [1, 32, 16, 1]
EPOCHS = 10
LEARINN_RATE = 0.1
SEED = 141
SHOW_PERCENTAGE = 1
BIAS = True

# endregion

# region Drawing Parameters

COMPUTE_LOSS = True
ITERATIONS = 20
STEP = 1
DRAW_GRAPH = False

# endregion

# endregion

# region Functions

def train_and_test(perceptron, train_dataset, test_dataset):
    if not COMPUTE_LOSS:
        perceptron.train(train_dataset, SHOW_PERCENTAGE)
        _, _, predictions = perceptron.test(test_dataset, SHOW_PERCENTAGE)
    else:
        predictions = train_test_draw_loss_function(perceptron, train_dataset, test_dataset)
    return predictions

def normalize_dataset(train_dataset, test_dataset):
    if PROBLEM_TYPE == problem_type.Regression:
        train_dataset, test_dataset = normalize(train_dataset, test_dataset)
    return (train_dataset, test_dataset)

def draw_regression(train_dataset, test_dataset, predictions):
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

def draw_classification(train_dataset, test_dataset, predictions):
    points = []
    train_points = []
    for i in range(len(test_dataset)):
        points.append((test_dataset[i][0], predictions[i]))
    for i in range(len(train_dataset)):
        train_points.append(
            (train_dataset[i][0], np.nonzero(train_dataset[i][1])))

    # generate_classification_graph_of_points(points, train_points)
    generate_classification_graph_for_model(perceptron, train_dataset, test_dataset)

def train_test_draw_loss_function(perceptron, train_dataset, test_dataset):
    print_flag = SHOW_PERCENTAGE != -1
    show_per = SHOW_PERCENTAGE
    if print_flag:
        print('----START TEST----')
    perceptron.epochs = STEP
    epochs=[]
    train_loss= []
    test_loss= []
    c = 0
    iter = ITERATIONS*STEP
    for i in range(iter):
        perceptron.train(train_dataset, -1)
        if i >= c:
            c += STEP
            epochs.append(i)
            test_loss.append(perceptron.test(train_dataset, -1)[1])
            prediction_rate, loss, predictions = perceptron.test(test_dataset, -1)
            train_loss.append(loss)

        if print_flag:
            per = i/iter * 100
            if per>= show_per:
                print(f'Training progress status: {round(i/iter * 100, 2)}%')
                show_per += SHOW_PERCENTAGE

    epo = EPOCHS - iter
    if epo > 0:
        perceptron.epochs = epo
        perceptron.train(train_dataset, SHOW_PERCENTAGE)
        prediction_rate, loss, predictions = perceptron.test(test_dataset, -1)

    if print_flag:
        print(f'Training progress status: {100}%')
        print('----TRAINING FINISHED----')
        if PROBLEM_TYPE == problem_type.Classification: print(f'Correct predicted rate: {prediction_rate}%')
        print(f'Loss function : {loss}')

    generate_loss_function_graph(epochs, train_loss, test_loss, str(LOSS_FUNCTION.__name__))
    return predictions

# endregion

if __name__ == "__main__":
    # Init perceptron and read datasets
    perceptron = MLP(PROBLEM_TYPE, LAYERS, ACTIVATION_FUNCTION, OUTPUT_FUNCTION, LOSS_FUNCTION, EPOCHS, LEARINN_RATE, SEED, BIAS)
    train_dataset = prepare_data(PROBLEM_TYPE, PATH_TO_TRAIN_DATASET)
    test_dataset = prepare_data(PROBLEM_TYPE, PATH_TO_TEST_DATASET)

    # Performe normalization if needed
    (train_dataset, test_dataset) = normalize_dataset(train_dataset, test_dataset)
    
    # Train and test (drawing loss function)
    predictions = train_and_test(perceptron, train_dataset, test_dataset)

    # Draw graphs
    if DRAW_GRAPH and PROBLEM_TYPE == problem_type.Regression:
        draw_regression(train_dataset, test_dataset, predictions)
    elif DRAW_GRAPH and PROBLEM_TYPE == problem_type.Classification:
        draw_classification(train_dataset, test_dataset, predictions)



  
