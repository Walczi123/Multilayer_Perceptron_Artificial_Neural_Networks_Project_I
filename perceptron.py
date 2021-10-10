import numpy as np
from typing import Callable

from activation_functions import sigmoid


class MLP:
    """
    Neural Network Class
    """

    def __init__(self, layers: list, activation_function: Callable, transfer_function: Callable, epochs: int, learning_rate: float, learning_coefficient: float, seed: int):
        """
        Args:
            layers (list): list of layers in the network
            activation_function (function): sigmoid function used in the internal neurons of the network
            transfer_function (function): activation function used on the output layer
            epochs (int): number of epochs to learn
            learning_rate (float): learning rate coefficient
            learning_coefficient (float): learning coefficient
            seed (int): number used as a seed for random number generator
        """
        self.layers = layers
        self.activation_function = activation_function
        self.transfer_function = transfer_function
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_coefficient = learning_coefficient

        # Init weights
        self.weights = []
        for i in range(len(layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i + 1])
            print("w", w,"\n")
            self.weights.append(w / np.sqrt(layers[i]))
        
        w = np.random.randn(layers[-2] + 1, layers[-1])
        print("w", w,"\n")
        self.weights.append(w / np.sqrt(layers[-2]))

        print(self.weights)

    def backpropagation(self):
        pass

    def feed_forward(self):
        x = self.activation_function(2)
        print(x)

    def train(self, data_set):
        for _ in range(self.epochs):
            delta_weights = 0

            for (X, Y) in data_set:
                output = self.feed_forward(X)
                delta_weights = self.backpropagation(output, Y)
                self.weights += delta_weights

    def predict(self):
        return self.feed_forward()

    def test(self):
        pass


if __name__ == "__main__":
    # perceptron = MLP([2, 2, 1], sigmoid, sigmoid, 1, 1, 1, 1)
    # perceptron.predict()

    a = [[1,2],[3,4],[5,6]]
    b = [1,2,3]
    print(np.dot(b,a))
