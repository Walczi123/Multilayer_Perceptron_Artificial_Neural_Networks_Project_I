from activation_functions import sigmoid

class MLP:
    """
    Neural Network Class
    """

    def __init__(self, layers:list, activation_function):
        self.activation_function = activation_function

    def backpropagation(self):
        pass

    def feed_forward(self):
        x = self.activation_function(2)
        print(x)

    def train(self):
        pass

    def predict(self):
        return self.feed_forward()

    def test(self):
        pass

if __name__ == "__main__":
    perceptron = MLP([2, 2, 1], sigmoid)
    perceptron.predict()