import numpy as np
from common.functions import function_type

class MLP:
    """
    Neural Network Class
    """
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
        np.random.seed(seed)
        self.layers = layers
        self.activation_function = activation_function
        self.transfer_function = transfer_function
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_coefficient = learning_coefficient

        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            if True:
                self.biases.append(np.random.randn(layers[i + 1]))
            w = np.random.randn(layers[i], layers[i + 1])
            self.weights.append(w / np.sqrt(layers[i]))

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
        tmp = error
        for layer in range(len(self.weights) - 2, -1, -1): # PYTANIE: wzór ? dobry?
            tmp = np.dot(tmp, self.weights[layer+1].T)
            tmp = np.multiply(tmp,self.activation_function.derivative(z[layer+1]))
            D_biases.append(tmp)
            tmp1 = np.dot(tmp.T, np.atleast_2d(outputs[layer]))
            D_weights.append(tmp1)

        D_weights = D_weights[::-1]
        D_biases = D_biases[::-1]

        for i in range(len(self.weights)): # PYTANIE: UPDATE wag przy propagacji wstecznej czy po przejściu przez wyszystkie wagi?
            self.weights[i] += -self.learning_rate * D_weights[i][0].T
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
    perceptron = MLP([2, 3,5,1], function_type.Sigmoid, function_type.Softmax, 1, 0.5, 0.5, 1)
    data_set = [[np.array([1,2]),1],[np.array([-1,-2]),0],[np.array([2,2]),1]]
    perceptron.train(data_set)
    print("-------")
    print(perceptron.predict(data_set[0][0]))