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

# def softmax(x):
#     exps = np.exp(x - np.max(x, keepdims=True))
#     return exps / np.sum(exps, keepdims=True)

def softmax(x):
    exps = np.exp(x)
    return exps / exps.sum()

def softmax_derivative(x):
    sx = softmax(x)
    return np.multiply(sx, 1-sx)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return np.array([relu_derivative_single(xi) for xi in x])

def relu_derivative_single(x):
    if x > 0:
        return 0
    return 1

# def cross_entropy(predictions, targets, epsilon=1e-12):
#     # predictions = np.clip(predictions, epsilon, 1. - epsilon)
#     predictions = np.array(predictions)
#     N = predictions.shape[0]
#     ce = -np.sum(targets*np.log(predictions+1e-9))/N
#     return ce
def cross_entropy(prediction, target):
    prediction = np.array(prediction)
    target = np.array(target)
    return np.mean(np.sum(-np.multiply(target, np.log(prediction + 1e-7))))

def mse(predicteds, targets):
    # targets = np.concatenate(targets)
    predicteds = np.concatenate(predicteds)
    diff = targets - predicteds
    square = np.square(diff)
    summation = sum(square)
    return summation / len(targets)

def hinge(predicteds, targets):
    # replacing 0 = -1
    new_predicted = np.array([-1 if i==2 else i for i in predicteds])
    new_actual = np.array([-1 if i==2 else i for i in targets])

    # calculating hinge loss
    hinge_loss = np.mean([max(0, 1-x*y) for x, y in zip(new_actual, new_predicted)])
    return hinge_loss

def hinge_f(prediction, target):
    prediction = np.array(prediction)
    target = np.array(target)
    Y_target_1 = np.argmax(target)
    N = prediction.shape[0]
    delta = 1.0  # A fixed margin
    # correct_scores = prediction[np.arange(N), Y_target_1]

    diffs = np.maximum(0, prediction - target + delta)
    # diffs[np.arange(N), Y_target_1] = 0
    loss = np.sum(diffs) / N

    return loss

def msle(predictions, targets):
    targets=np.array(targets)
    predictions = np.concatenate(predictions)
    min = predictions.min()
    if min < 0:
        predictions = predictions - (2*min)
        targets = targets - (2*min)
    log_t = np.log(targets)
    log_p = np.log(predictions)
    diff =  log_t - log_p
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
    Relu = relu
    Relu.derivative = relu_derivative
    Cross_entropy = cross_entropy
    MSE = mse
    MSLE = msle
    Hinge = hinge
    
