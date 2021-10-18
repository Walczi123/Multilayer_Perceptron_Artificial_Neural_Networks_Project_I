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
        self.biases = []
        for i in range(len(layers)-1):
            if True:
                self.biases.append(np.random.randn(layers[i + 1]))
            w = np.random.randn(layers[i], layers[i + 1])
            self.weights.append(w / np.sqrt(layers[i]))
        

        # print(self.weights)

    # def backpropagation(self, outputs, result):
    #     error = outputs[-1] - result
    #     D = [error * self.sigmoid_deriv(outputs[-1])]

    #     for layer in np.arange(len(outputs) - 2, 0, -1):
    #         delta = D[-1].dot(self.W[layer].T)
    #         delta = delta * self.sigmoid_deriv(outputs[layer])
    #         D.append(delta)

    #     print("self.weights",self.weights)
    #     self.alpha = 0.1

    def feed_forward(self, data):
        output=[data]
        z = [data]
        tmp = data

        for i in range(len(self.weights)):
            tmp = np.dot(tmp, self.weights[i])
            if True:               
                tmp += self.biases[i]
            z.append(np.array(tmp))
            tmp = np.array([self.activation_function(x) for x in tmp])
            output.append(tmp)
        return output, z

    def backpropagation(self, outputs, z, result, data):
        error = outputs[-1] - np.atleast_2d(result)
        D_weights = [np.dot(error.T, [outputs[-2]])]
        D_biases = [error]
        print("D",  D_weights)
        tmp = error
        for layer in range(len(self.weights) - 2, -1, -1): # PYTANIE: wzór ? dobry?
            print("layer", layer)
            print("weigh", self.weights[layer+1])
            print("T", tmp)
            tmp = np.dot(tmp, self.weights[layer+1].T)
            print("W_T", tmp)
            print("DEriv",f_deriv(z[layer+1]) )
            tmp = np.multiply(tmp,f_deriv(z[layer+1]))
            print("W_T deriv", tmp)
            D_biases.append(tmp)
            print("OUT", outputs[layer])
            tmp1 = np.dot(tmp.T, np.atleast_2d(outputs[layer]))
            D_weights.append(tmp1)

        D_weights = D_weights[::-1]
        D_biases = D_biases[::-1]

        for i in range(len(self.weights)): # PYTANIE: UPDATE wag przy propagacji wstecznej czy po przejściu przez wyszystkie wagi?
            print("i", i)
            print("self.weights[i]",  self.weights[i])
            print("D_weights[i]", D_weights[i])
            print("self.biases[i]",  self.biases[i])
            print("D_biases[i]", D_biases[i])
            self.weights[i] += -self.learning_rate * D_weights[i].T
            self.biases[i] += -self.learning_rate * D_biases[i][0]

    def train(self, dataset, show_percentage = 1):
        print('----START TRAINING----')
        showing_param = 0
        for i in range(self.epochs):
            for X, Y in dataset:
                output, z = self.feed_forward(X)
                self.backpropagation(output, z, Y, X)
            if i/self.epochs >= showing_param/100:
                print(f'Training progress status: {showing_param}%')
                showing_param += show_percentage
        print(f'Training progress status: {100}%')
        print('----TRAINING FINISHED----')

    def predict(self, data):
        return self.feed_forward(data)[0][-1]

    def test(self, dataset, show_percentage = 1):
        print('----START TEST----')
        counter = 0
        showing_param = 0
        len_dataset = len(dataset)
        for i in range(len_dataset-1):
            data, result = dataset[i]
            prediction = self.predict(data)
            if i/len_dataset >= showing_param/100:
                print(f'Test progress status: {showing_param}%')
                showing_param += show_percentage
            if prediction == result:
                counter += 1
        print(f'Test progress status: {100}%')
        print('----TEST FINISHED----')
        print(f'Correct predicted rate: {counter/len_dataset * 100}%')


if __name__ == "__main__":
    perceptron = MLP([2, 5, 20, 3, 5,10,3], sigmoid, sigmoid, 1, 0.5, 0.5, 1)
    data_set = [[np.array([1,2]),1],[np.array([-1,-2]),0],[np.array([2,2]),1]]
    perceptron.train(data_set)
    print("-------")
    print(perceptron.predict(data_set[0][0]))
    # a = np.array([[1,2,3]])
    # b = np.array([[1,2,3]])
    # print(np.dot(a.T,b))