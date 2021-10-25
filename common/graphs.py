import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from common.problem_type import problem_type

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


def generate(model, dataset, test_dataset):
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
    grid = np.hstack((r1, r2))
    # define the model
    # fit the model
    model.train(dataset)
    # make predictions for the grid
    # yhat = model.predict(grid)
    _, _, predictions = model.test(test_dataset)
    # reshape the predictions back into a grid
    zz = predictions
    # plot the grid of x, y and z values as a surface
    pyplot.contourf(xx, yy, zz, cmap='Paired')
    # create scatter plot for samples from each class
    unique = np.unique(y)
    for class_value in range(len(unique)):
        row_ix = np.where(y == class_value+1)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')

    pyplot.show()


def generate_regression_graph(targets, predictions, train):
    pyplot.scatter(targets[0], targets[1], c="black")
    pyplot.scatter(predictions[0], predictions[1], c="red")
    pyplot.scatter(train[0], train[1], c="green")
    pyplot.show()


if __name__ == "__main__":
    dataset = prepare_data(problem_type.Classification,
                           "data/classification/data.simple.test.100.csv")
    generate_classification_graph_of_points(dataset)
    # generate_regression_graph()
