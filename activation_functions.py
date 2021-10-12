import math
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def tanh(x):
  return np.tanh(x)

def simple(x):
  return x