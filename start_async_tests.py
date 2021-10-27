import multiprocessing
import time
from common.tests.Test import Test
from common.functions import function_type
from perceptron import MLP
from common.problem_type import problem_type
from common.reader import prepare_data

SEED = 12020122
REPETITIONS = 50


def generate_instances():
    result = []
    
    mpl = MLP([1,2,1], function_type.Sigmoid, function_type.Sigmoid, 1, 0.1, 0.1, SEED, True)
    train_dataset = prepare_data(problem_type.Regression, "data/regression/data.cube.train.100.csv", True)
    test_dataset = prepare_data(problem_type.Regression, "data/regression/data.cube.test.100.csv", True)
    result.append(Test(mpl, train_dataset, test_dataset, n_repetition=REPETITIONS,
                       name="mpl121", seed=SEED))

    # warsty 
    4 x rozne warsty x 2 rozne problemy x 2 bias x 3 funkcje aktywacji (sigmoid, tanh, identity/gauss) = 156
    [1,1] [1,2,1], [1,4,2,1],[1,8,4,2,1], [1,16,8,4,2,1]
    [1,8,1], [1,16,8,1],[1,32,16,8,1], [1,64,32,16,8,1]
    [1,32,1], [1,64,32,1],[1,128,64,32,1], [1,256,128,64,32,1]
    
    # biasy
    # funkcje aktywacji 3
    
    ile iteracji = 1000


    datasety (8) x 2 licznosci (1000 i 100) = 16

    wykresy:
    - wykresy błedów (2)
    - wykresy wynikowe 
    czyli 3 wykresy


    nazwa pliku:
    Rodzaj probleu - ilośc warst - ilośc nodów - biases - funkcja aktywacji - dataset - licznośc datasetu -(wykres błedu/wynikowy).csv

    return result


def run_test(test):
    print("in t: " + test.name)
    test.start()


def run_tests():
    iterable = generate_instances()
    # print("iterable", iterable)
    start_time = time.time()

    # p = multiprocessing.Pool()
    # p.map_async(run_test, iterable)

    # p.close()
    # p.join()
    iterable[0].start()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    run_tests()