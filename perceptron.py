import numpy as np
# from typing import Callable

from activation_functions import sigmoid, simple


def f_deriv(x):
    return x * (1 - x)

class MLP:
    """
    Neural Network Class
    """

    # def __init__(self, layers:list, activation_function: Callable, transfer_function: Callable, epochs: int, learning_rate: float, learning_coefficient: float, seed: int):
    def __init__(self, layers, activation_function, transfer_function, epochs, learning_rate, learning_coefficient, seed):

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
            # w = np.random.randn(layers[i] + 1, layers[i + 1])
            w=[]
            for _ in range(layers[i] + 1):
                w.append([1 for _ in range(layers[i + 1])])
            print("w", w,"\n")
            # self.weights.append(w / np.sqrt(layers[i]))
            self.weights.append(w)
        
        w = np.random.randn(layers[-2] + 1, layers[-1])
        # print("w", w,"\n")
        self.weights.append(w / np.sqrt(layers[-2]))

        # print(self.weights)

    def backpropagation(self, outputs, result):
        error = outputs[-1] - result
        D = [error * self.sigmoid_deriv(outputs[-1])]

        for layer in np.arange(len(outputs) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(outputs[layer])
            D.append(delta)

        print("self.weights",self.weights)
        self.alpha = 0.1

    def feed_forward(self, data):
        output=[np.append(data, 1)]
        # print("data",data)
        tmp = data
        # tmp = data     
        for weight_matrix in self.weights:
            # print("tmp1",tmp)
            # tmp.insert(1) # bias
            tmp = np.append(tmp, 1) # bias
            # print("tmp2",tmp)
            # print("in tmp", tmp)
            # print("weight_matrix", weight_matrix)
            # tmp = np.dot(weight_matrix, tmp)
            tmp = np.dot(tmp, weight_matrix)
            tmp = np.array([self.activation_function(x) for x in tmp])
            # print("out tmp", tmp) 
            output.append(tmp)
        return output

    def backpropagation(self, outputs, result):
        print("outputs", outputs)
        print("outputs[-1]", outputs[-1])
        error = outputs[-1] - result #czemu -2? a nie -1? 
        print("error",error)
        D = [error * f_deriv(outputs[-1])]
        print("D1",D)

        for layer in np.arange(len(outputs)-3, 1, -1):
            print("layer1", layer)
            print("D[-1]",D[-1])
            print("self.weights[layer]",self.weights[layer])
            delta = D[-1].dot(np.atleast_2d(self.weights[layer]).T)
            print("delta1", delta)
            print("outputs[layer]",outputs[layer])
            delta = delta[:-1] * f_deriv(outputs[layer])
            D.append(delta)

        D = D[::-1]
        print("D2",D)
        for layer in np.arange(0, len(self.weights)):
            print("layer2", layer)
            print("outputs["+str(layer)+"]", outputs[layer])
            print("D["+str(layer)+"]", D[layer])
            print("weights[["+str(layer)+"]", self.weights[layer])
            k = outputs[layer].T.dot(D[layer])
            print("k",k)
            # print("weights[["+str(layer)+"]", self.weights[layer])
            self.weights[layer] += -self.learning_rate  * k

    def train(self, data_set):
        for _ in range(self.epochs):
            delta_weights = 0

            for X, Y in data_set:
                output = self.feed_forward(X)
                delta_weights = self.backpropagation(output, Y)
                # self.weights += delta_weights

    def predict(self, data):
        return self.feed_forward(data)[-1]

    def test(self):
        pass


if __name__ == "__main__":
    perceptron = MLP([2, 3, 1], sigmoid, sigmoid, 1, 0.5, 0.5, 1)
    data_set = [[np.array([1,2]),1],[np.array([-1,-2]),0],[np.array([2,2]),1]]
    perceptron.train(data_set)
    print("-------")
    print(perceptron.predict(data_set[0][0]))

    # a = [[1,2],[3,4],[5,6]]
    # b = [1,2,3]
    # print(np.dot(b,a))
