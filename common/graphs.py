import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from common.problem_type import problem_type
from numpy import hstack

from common.reader import prepare_data

first_colors = ['darkred', 'darkgreen', 'darkblue']
second_colors = ['lightcoral', 'lime', 'royalblue']


def generate_classification_graph_of_points(dataset, training):
    X = np.array([data[0] for data in training])
    y = np.array([data[1] for data in training])

    unique = np.unique(y)
    for class_value in range(len(unique)):
        row_ix = np.where(y == class_value + 1)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1],
                       color=first_colors[class_value])

    X = np.array([data[0] for data in dataset])
    y = np.array([data[1] for data in dataset])

    unique = np.unique(y)
    for class_value in range(len(unique)):
        row_ix = np.where(y == class_value + 1)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1],
                       color=second_colors[class_value])

    pyplot.show()


def generate_classification_graph_for_model(model12, dataset, test_dataset):
    # generate dataset
    # X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
    X = np.array([data[0] for data in dataset])
    y = np.array([data[1] for data in dataset])
    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = hstack((r1,r2))
    # define the model
    model = LogisticRegression()
    # fit the model
    model.fit(X, y)
    # make predictions for the grid
    yhat = model.predict(grid)
    pre = model12.predict_list(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    pyplot.contourf(xx, yy, zz, cmap='Paired')
    # create scatter plot for samples from each class
    unique = np.unique(y)
    for class_value in range(len(unique)):
        row_ix = np.where(y == class_value)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')

    pyplot.show()


def generate_regression_graph(targets, predictions, train):
    pyplot.scatter(targets[0], targets[1], c="black")
    pyplot.scatter(predictions[0], predictions[1], c="red")
    pyplot.scatter(train[0], train[1], c="green")
    pyplot.show()


def generate_loss_function_graph(epochs, train_loss, test_loss):
    # pyplot.scatter(epochs, train_loss, s=0.1, c="black")
    # pyplot.scatter(epochs, test_loss, s=0.1, c="red")
    # pyplot.show()

    pyplot.plot(epochs, train_loss, 'g', label='Training accuracy')
    pyplot.plot(epochs, test_loss, 'b', label='validation accuracy')
    pyplot.title('Training and Validation accuracy')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    dataset = prepare_data(problem_type.Classification,
                           "data/classification/data.simple.test.100.csv")
    generate_classification_graph_of_points(dataset)
    # generate_regression_graph()
