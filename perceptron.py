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

        self.weights = None

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
    perceptron = MLP([2, 2, 1], sigmoid, sigmoid, 1, 1, 1, 1)
    perceptron.predict()
