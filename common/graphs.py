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


def generate_classification_graph_for_modelv2(model12, dataset, test_dataset):
    X = np.array([data[0] for data in dataset])
    y = np.array([data[1] for data in dataset])
    min1, max1 = X[:, 0].min(), X[:, 0].max()
    min2, max2 = X[:, 1].min(), X[:, 1].max()

    dim1_coeff = (max1 - min1) * 0.1
    dim2_coeff = (max2 - min2) * 0.1

    min1, max1 = min1 - dim1_coeff, max1 + dim1_coeff
    min2, max2 = min2 - dim2_coeff, max2 + dim2_coeff
    x1grid = np.arange(min1, max1, 0.05)
    x2grid = np.arange(min2, max2, 0.05)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = hstack((r1,r2))
    pre = np.array(model12.predict_list(grid))
    zz = pre.reshape(xx.shape)
    len_unique = len(np.unique(y))
    for class_value in range(len_unique):
        row_ix = np.where(zz == class_value+1)
        pyplot.scatter(xx[row_ix], yy[row_ix], s=75, color = second_colors[class_value])

    for class_value in range(len_unique):
        row_ix = np.where(y == class_value+1)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label = str(class_value+1), color = first_colors[class_value])

    pyplot.legend()
    pyplot.show()

def generate_classification_graph_for_model(model12, dataset, test_dataset):
    X = np.array([data[0] for data in dataset])
    y = np.array([data[1] for data in dataset])
    min1, max1 = X[:, 0].min(), X[:, 0].max()
    min2, max2 = X[:, 1].min(), X[:, 1].max()

    dim1_coeff = (max1 - min1) * 0.1
    dim2_coeff = (max2 - min2) * 0.1

    min1, max1 = min1 - dim1_coeff, max1 + dim1_coeff
    min2, max2 = min2 - dim2_coeff, max2 + dim2_coeff

    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = hstack((r1,r2))

    pre = np.array(model12.predict_list(grid))
    zz = pre.reshape(xx.shape)

    len_unique = len(np.unique(y))
    pyplot.contourf(xx, yy, zz, levels=len_unique-1, colors = second_colors)

    for class_value in range(len_unique):
        row_ix = np.where(y == class_value+1)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label = str(class_value+1), color = first_colors[class_value])

    pyplot.legend()
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
