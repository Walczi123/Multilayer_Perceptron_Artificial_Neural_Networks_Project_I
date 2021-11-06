import ssl
import time
from tensorflow.keras.datasets import mnist
# from tensorflow import keras
from matplotlib import pyplot
from common.functions import function_type
from common.problem_type import problem_type
from perceptron import MLP
import numpy as np
import requests
requests.packages.urllib3.disable_warnings()


def invert_images(images):
    for img in images:
        img = 255 - img
    return images


def threshold_images(images, threshold):
    for img in images:
        img[img > threshold] = 255
        img[img <= threshold] = 0
    return images


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

TAKE_PART = 100
INVERT = False
THRESHOLD = True
THRESHOLD_VALUE = 127

if INVERT:
    train_X = invert_images(train_X)
    test_Y = invert_images(test_X)

if THRESHOLD:
    train_X = threshold_images(train_X, THRESHOLD_VALUE)
    test_X = threshold_images(test_X, THRESHOLD_VALUE)


train_dataset = []
# r = len(train_X)
r_train = int(len(train_X)/TAKE_PART)
for idx in range(r_train):
    # for idx in range(300):
    train_dataset.append([np.array(train_X[idx].flatten()), train_y[idx]])
test_dataset = []
# r = len(test_X)
r_test = int(len(test_X)/TAKE_PART)
for idx in range(r_test):
    # for idx in range(10):
    test_dataset.append([np.array(test_X[idx].flatten()), test_y[idx]])

image_len = 28 * 28
classes_no = unique = len(np.unique(train_y))
PROBLEM_TYPE = problem_type.Classification
ACTIVATION_FUNCTION = function_type.Sigmoid
OUTPUT_FUNCTION = function_type.Softmax
LOSS_FUNCTION = function_type.Cross_entropy
# LAYERS = [image_len, classes_no]
LAYERS = [image_len, 700, 500, classes_no]
EPOCHS = 20
LEARINN_RATE = 0.001
SEED = 1231
SHOW_PERCENTAGE = 0.01
BIAS = True


def save_to_file(rate):
    timestr = time.strftime("%d_%m_%Y-%H_%M-%S")
    f = open(f"minst/MNIST_{timestr}", "w")
    f.write(f"Rate: {rate}\n")
    f.write(f"Layers: {LAYERS}\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Learning rate: {LEARINN_RATE}\n")
    f.write(f"Activation function: {ACTIVATION_FUNCTION.__name__}\n")
    f.write(f"Output function: {OUTPUT_FUNCTION.__name__}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Inverted: {INVERT}")
    f.write(f"Thresholded: {THRESHOLD}")
    f.write(f"Threshold_value: {THRESHOLD_VALUE}")
    if TAKE_PART != 1:
        f.write(f"Part of dataset: {TAKE_PART}\n")
    f.close


if __name__ == "__main__":
    perceptron = MLP(PROBLEM_TYPE, LAYERS, ACTIVATION_FUNCTION,
                     OUTPUT_FUNCTION, LOSS_FUNCTION, EPOCHS, LEARINN_RATE, SEED, BIAS)

    perceptron.train(train_dataset, SHOW_PERCENTAGE, category_shift=0)
    rate, _, _ = perceptron.test(
        test_dataset, SHOW_PERCENTAGE, category_shift=0)

    # for i in range(30):
    #     test_case = perceptron.predict(test_X[500  + i].flatten())
    #     pyplot.title(label=f"Label: {test_y[500 + i]} - Predicted: {test_case}")
    #     pyplot.imshow(test_X[500 + i], cmap=pyplot.get_cmap('gray'))
    #     pyplot.show()

    # print(train_X[0])
    # print(train_X[0].shape)
    # print(train_X[0].flatten())
    # print(train_X[0].flatten().shape)
    # print(len(train_X))
    # print(len(train_y))
    # print(len(test_X))
    # print(len(test_y))
    # print(len(train_dataset))
    # print(len(test_dataset))
    save_to_file(rate)
