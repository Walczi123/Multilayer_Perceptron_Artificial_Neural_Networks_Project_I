import multiprocessing
import time
import itertools
import tqdm
from common.tests.Test import Test
from common.functions import function_type
from perceptron import MLP
from common.problem_type import problem_type
from common.reader import prepare_data
import numpy as np

SEED = 12020122
REPETITIONS = 60 # 60
EPOCHS = 5 # 5
ITERATIONS = REPETITIONS * EPOCHS # = 300
LEARINN_RATE = 0.1

def layer_amount(layers):
    return str(len(layers))

def nodes_amount(layers):
    return str(int(np.sum(layers)))

def dataset_name_amount(dataset_path):
    splited = dataset_path.split('.')
    return f'{splited[1].replace("_","-")}_{splited[3]}'

def get_input_output_dataset(problem_type, dataset_path):
    if problem_type == problem_type.Regression:
        return (1,1)
    else:
        dataset = prepare_data(problem_type, dataset_path)
        len_test_inputs = len(np.unique([y for x, y in dataset]))
        return (2, len_test_inputs)

def generate_instances():
    result = []
    # region Parameters
    # layers = [[1,1], 
    #           [1,2,1],  [1,4,2,1],   [1,8,4,2,1],     [1,16,8,4,2,1],
    #           [1,8,1],  [1,16,8,1],  [1,32,16,8,1],   [1,64,32,16,8,1],
    #           [1,32,1], [1,64,32,1], [1,128,64,32,1], [1,256,128,64,32,1]]

    hidden_layers = [[], 
              [2],  [4,2],   [8,4,2],     [16,8,4,2],
              [8],  [16,8],  [32,16,8],   [64,32,16,8],
              [32], [64,32], [128,64,32], [256,128,64,32]]

    problems  = [problem_type.Classification, problem_type.Regression]
    classification_loss_function = [function_type.Cross_entropy, function_type.Hinge]
    regression_loss_function     = [function_type.MSE,           function_type.MSLE]

    biases = [True, False]
    activation_functions = [function_type.Sigmoid, function_type.Tanh] #jeszcze jedna identity/gauss
    output_functions = [function_type.Softmax, function_type.Indentity]

    # Regression
    datasets_regression = [
    ['data/regression/data.cube.train.100.csv',        'data/regression/data.cube.test.100.csv'], #cube100
    ['data/regression/data.cube.train.1000.csv',       'data/regression/data.cube.test.1000.csv'], #cube1000
    ['data/regression/data.activation.train.100.csv',  'data/regression/data.activation.test.100.csv'], #activation100
    ['data/regression/data.activation.train.1000.csv', 'data/regression/data.activation.test.1000.csv'] #activation1000
    ] 

    #Classification
    datasets_classification = [
    ['data/classification/data.simple.train.100.csv',       'data/classification/data.simple.test.100.csv'], #simple100
    ['data/classification/data.simple.train.1000.csv',      'data/classification/data.simple.test.1000.csv'], #simple1000
    ['data/classification/data.three_gauss.train.100.csv',  'data/classification/data.three_gauss.test.100.csv'], #three_gauss100
    ['data/classification/data.three_gauss.train.1000.csv', 'data/classification/data.three_gauss.test.1000.csv'] #three_gauss1000
    ]

    # endregion 
    # region Init instances

    problems_loss_functions_datasets=[]

    for r in datasets_classification: #Classification
        problems_loss_functions_datasets.append((problems[0], classification_loss_function, r, get_input_output_dataset(problems[0],r[0]), output_functions[0]))

    for r in datasets_regression: #Regression
        problems_loss_functions_datasets.append((problems[1], regression_loss_function, r, get_input_output_dataset(problems[1],r[0]), output_functions[1]))
    
    
    for r in itertools.product(problems_loss_functions_datasets, hidden_layers, activation_functions, biases):
        layer = r[1].copy()
        layer.insert(0, r[0][3][0])
        layer.append(r[0][3][1])
        # problem_type, layers, activation_function, output_function, loss_function, epochs, learning_rate, seed, bias=False
        mpl = MLP(r[0][0], layer, r[2], r[0][4], None, EPOCHS, LEARINN_RATE, SEED, r[3])
        # mpl, train_dataset_path, test_dataset_path, loss_function1, loss_function2, n_repetition=1, name="test", seed=None):
        result.append(Test(mpl, r[0][2][0], r[0][2][1], r[0][1][0], r[0][1][1], n_repetition=REPETITIONS, seed=SEED,
            name=f'{r[0][0].__name__}_{layer_amount(r[1])}_{nodes_amount(r[1])}_{SEED}_{1 if r[3] else 0}_{r[2].__name__}_{dataset_name_amount(r[0][2][0])}'))
        # Rodzaj probleu - ilośc warst - ilośc nodów - biases - funkcja aktywacji - dataset - licznośc datasetu -(wykres błedu/wynikowy).csv

    expeded_len = (len(hidden_layers) * len(biases) * len(activation_functions)) * (len(datasets_regression) + len(datasets_classification))
    assert len(result) == expeded_len, f'Incorrect amount of test cases ({len(result)} != {expeded_len})'

    # endregion
    return result


def run_test(test):
    print(f'start of {test.name}')
    test.start()


def run_tests():
    iterable = generate_instances()[:10]
    start_time = time.time()

    p = multiprocessing.Pool()
    for _ in tqdm.tqdm(p.imap_unordered(run_test, iterable), total=len(iterable)):
        pass
    # p.map_async(run_test, iterable)

    p.close()
    p.join()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    run_tests()