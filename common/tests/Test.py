import numpy as np
from datetime import datetime
import time
from common.reader import normalize, prepare_data
from common.problem_type import problem_type
from common.graphs import draw_regression, generate_classification_graph_for_model, generate_loss_function_graph, generate_regression_graph


class Test:
    def __init__(self, mpl, train_dataset_path, test_dataset_path, loss_function1, loss_function2, n_repetition=1, name="test", seed=None):
        self.mpl = mpl
        self.n_repetition = n_repetition
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.train_dataset = None
        self.test_dataset = None
        self.loss_function1 = loss_function1
        self.loss_function2 = loss_function2
        self.name = name
        self.seed = seed

    def get_data(self):
        self.train_dataset = prepare_data(self.mpl.problem_type, self.train_dataset_path)
        self.test_dataset = prepare_data(self.mpl.problem_type, self.test_dataset_path)
        if self.mpl.problem_type == problem_type.Regression:
            self.train_dataset, self.test_dataset = normalize(self.train_dataset, self.test_dataset)
        test_targets = []
        test_inputs = []
        for i in range(len(self.test_dataset)):
            test_targets.append(self.test_dataset[i][1])
            test_inputs.append(self.test_dataset[i][0])
        train_targets = []
        train_inputs = []
        for i in range(len(self.train_dataset)):
            train_targets.append(self.train_dataset[i][1])
            train_inputs.append(self.train_dataset[i][0])
        return train_inputs, train_targets, test_inputs, test_targets

    def classification_accuracy(self, predictions, test_targets):
        return np.sum(np.equal(np.array(predictions), np.array(test_targets)))/len(predictions) * 100


    def start(self):
        results = []
        np.random.seed(self.seed)
        train_inputs, train_targets, test_inputs, test_targets = self.get_data()
        for i in range(self.n_repetition):
            self.mpl.train(self.train_dataset, -1)
            predictions = self.mpl.predict_list(test_inputs)
            loss1 = self.loss_function1(predictions, test_targets)
            loss2 = self.loss_function2(predictions, test_targets)
            t_predictions = self.mpl.predict_list(train_inputs)
            t_loss1 = self.loss_function1(t_predictions, train_targets)
            t_loss2 = self.loss_function2(t_predictions, train_targets)
            if self.mpl.problem_type == problem_type.Regression:
                results.append(((i+1)*self.mpl.epochs, loss1, loss2, t_loss1, t_loss2))
            else:
                accuracy = self.classification_accuracy(predictions, test_targets)
                results.append(((i+1)*self.mpl.epochs, loss1, loss2, t_loss1, t_loss2, accuracy))
        # print("saving to file")
        # timestr = time.strftime("%d_%m_%Y-%H_%M-%S")
        self.save_to_file(results, f'data/results/{self.name}.csv')
        self.draw_graphs(results, predictions, 'data/results/graphs/')



    def save_to_file(self, results, file_path):
        f = open(file_path, "w")
        print("saving", self.name)
        f.write(f"Train dataset: {self.train_dataset_path}\n")
        f.write(f"Test dataset: {self.test_dataset_path}\n")
        f.write(f"Layers: {self.mpl.layers}\n")
        f.write(f"Test dataset: {1 if self.mpl.bias else 0}\n")
        f.write(f"Seed: {self.seed}\n")
        f.write(f"Learning rate: {self.mpl.learning_rate}\n")
        f.write(f"Activation function: {self.mpl.activation_function.__name__}\n")
        f.write(f"Output function: {self.mpl.output_function.__name__}\n")
        f.write(f"Problem type: {self.mpl.problem_type.__name__}\n")
        f.write(f"Epochs: {self.mpl.epochs * self.n_repetition}\n")
        if self.mpl.problem_type == problem_type.Regression:
            f.write(f"Epochs;Loss ({self.loss_function1.__name__});Loss ({self.loss_function2.__name__})\n")
            results = [f'{str(results[i][0])};{str(results[i][1])};{str(results[i][2])}\n'
                    for i in range(len(results))]
        else:
            f.write(f"Epochs;Loss ({self.loss_function1.__name__});Loss ({self.loss_function2.__name__});Accuracy\n")
            results = [f'{str(results[i][0])};{str(results[i][1])};{str(results[i][2])};{str(results[i][5])}\n'
                    for i in range(len(results))]
        f.writelines(results)
        f.close

    def draw_graphs(self, results, predictions, path):
        epochs = []
        train_loss1 = []
        train_loss2 = []
        test_loss1 = []
        test_loss2 = []
        for i in range(len(results)):
            epochs.append(results[i][0])
            train_loss1.append(results[i][3])
            train_loss2.append(results[i][4])
            test_loss1.append(results[i][1])
            test_loss2.append(results[i][2])
        generate_loss_function_graph(epochs, train_loss1, test_loss1, self.loss_function1.__name__, f'{path}{self.name}_loss1.png')
        generate_loss_function_graph(epochs, train_loss2, test_loss2, self.loss_function2.__name__, f'{path}{self.name}_loss2.png')
        if self.mpl.problem_type == problem_type.Regression:
            draw_regression(self.train_dataset, self.test_dataset, predictions, f'{path}{self.name}_result.png')
        else:
            generate_classification_graph_for_model(self.mpl, self.train_dataset, self.test_dataset, f'{path}{self.name}_result.png')
