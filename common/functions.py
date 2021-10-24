import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def simple(x):
    return x


def simple_derivative(x):
    return 1

# def softmax(x):
#    exp = np.exp(x)
#    return exp/sum(exp)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def softmax_derivative(x):
    sx = softmax_f(x)
    return np.multiply(sx, 1-sx)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2


class function_type():
    Sigmoid = sigmoid
    Sigmoid.derivative = sigmoid_derivative
    Simple = simple
    Simple.derivative = simple_derivative
    Softmax = softmax
    Softmax.derivative = softmax_derivative
    Tanh = tanh
    Tanh.derivative = tanh_derivative
    # Custom = None
    # Custom.derivative = None

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    # ce = -np.sum(targets*np.log(predictions+1e-9))/N
    ce = -np.sum(targets*np.log(predictions))/N
    return ce

def mse(predicteds, targets):
    targets = np.concatenate(targets)
    predicteds = np.concatenate(predicteds)
    diff = targets - predicteds #diff = [1, 2, 2]
    square = np.square(diff) # square = [1, 4, 4]
    summation = sum(square) # summation = 9
    return summation / len(targets) #3 