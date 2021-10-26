import ssl
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from matplotlib import pyplot
from common.functions import function_type
from common.problem_type import problem_type
from perceptron import MLP
import numpy as np
import requests
requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# train_X - list with MNIST training images
# train_y - list of labels of MNIST training images
# test_X - list with MNIST testing images
# test_y - list of labels of MN IST testing images
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_dataset = []
# for idx in range(len(train_X)):
for idx in range(300):
    train_dataset.append([np.array(train_X[idx].flatten()), train_y[idx]])
test_dataset = []
# for idx in range(len(test_X)):
for idx in range(10):
    test_dataset.append([np.array(test_X[idx].flatten()), test_y[idx]])

image_len = 28 * 28
classes_no = unique = len(np.unique(train_y))
PROBLEM_TYPE = problem_type.Classification
ACTIVATION_FUNCTION = function_type.Sigmoid
OUTPUT_FUNCTION = function_type.Softmax
LOSS_FUNCTION = function_type.Cross_entropy
LAYERS = [image_len, 256, 128, classes_no]
EPOCHS = 300
LEARINN_RATE = 5
SEED = 141
SHOW_PERCENTAGE = 1
BIAS = True

if __name__ == "__main__":
    perceptron = MLP(PROBLEM_TYPE, LAYERS, ACTIVATION_FUNCTION,
                     OUTPUT_FUNCTION, LOSS_FUNCTION, EPOCHS, LEARINN_RATE, SEED, BIAS)

    perceptron.train(train_dataset, SHOW_PERCENTAGE, category_shift=0)
    _, _, predictions = perceptron.test(
        test_dataset, SHOW_PERCENTAGE, category_shift=0)

    for i in range(30):
        test_case = perceptron.predict(test_X[500  + i].flatten())
        pyplot.title(label=f"Label: {test_y[500 + i]} - Predicted: {test_case}")
        pyplot.imshow(test_X[500 + i], cmap=pyplot.get_cmap('gray'))
        pyplot.show()

    # print(train_X[0])
    # print(train_X[0].shape)
    # print(train_X[0].flatten())
    # print(train_X[0].flatten().shape)
    # print(len(train_X))
    # print(len(train_y))
