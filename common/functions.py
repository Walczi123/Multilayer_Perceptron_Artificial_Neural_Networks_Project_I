import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def indentity(x):
    return x

def indentity_derivative(x):
    return 1

def softmax(x):
    exps = np.exp(x - np.max(x, keepdims=True))
    return exps / np.sum(exps, keepdims=True)


def softmax_derivative(x):
    sx = softmax(x)
    return np.multiply(sx, 1-sx)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce

def mse(predicteds, targets):
    # targets = np.concatenate(targets)
    predicteds = np.concatenate(predicteds)
    diff = targets - predicteds
    square = np.square(diff)
    summation = sum(square)
    return summation / len(targets)

class function_type():
    Sigmoid = sigmoid
    Sigmoid.derivative = sigmoid_derivative
    Indentity = indentity
    Indentity.derivative = indentity_derivative
    Softmax = softmax
    Softmax.derivative = softmax_derivative
    Tanh = tanh
    Tanh.derivative = tanh_derivative
    Cross_entropy = cross_entropy
    MSE = mse
