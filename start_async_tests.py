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
    
    mpl = MLP([1,2,1], function_type.Sigmoid, function_type.Sigmoid, 2, 0.1, 0.1, SEED, True)
    train_dataset = prepare_data(problem_type.Regression, "data/regression/data.cube.train.100.csv", True)
    test_dataset = prepare_data(problem_type.Regression, "data/regression/data.cube.test.100.csv", True)
    result.append(Test(mpl, train_dataset, test_dataset, n_repetition=REPETITIONS,
                       name="mpl121", seed=SEED,))

    return result


def run_test(test):
    print("in t: " + test.name)
    test.start()


def run_tests():
    iterable = generate_instances()
    # print("iterable", iterable)
    start_time = time.time()

    p = multiprocessing.Pool()
    p.map_async(run_test, iterable)

    p.close()
    p.join()
    # iterable[0].start()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    run_tests()