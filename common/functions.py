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
   