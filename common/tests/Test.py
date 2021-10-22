import numpy as np
from datetime import datetime
import time


class Test:
    def __init__(self, mpl, train_dataset, test_dataset, n_repetition=1, name="test", seed=None):
        self.mpl = mpl
        self.n_repetition = n_repetition
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.name = name
        self.seed = seed

    def start(self):
        results = []
        np.random.seed(self.seed)
        for i in range(self.n_repetition):
            self.mpl.train(self.train_dataset[0])
            _, loss = self.mpl.test(self.test_dataset[0])
            results.append(((i+1)*self.mpl.epochs, loss))
        # print("saving to file")
        timestr = time.strftime("%d_%m_%Y-%H_%M-%S")
        self.save_to_file(results, "data/results/" +
                          self.name + "_" +
                          str(self.n_repetition) + "_" + timestr + ".txt")


    def save_to_file(self, results, file_path):
        f = open(file_path, "w")
        print("saving", self.name)
        f.write(f"Train dataset: {self.train_dataset[1]}\n")
        f.write(f"Test dataset: {self.test_dataset[1]}\n")
        bias_str = "with biases" if self.mpl.bias else "without biases"
        f.write(f"Layers: {self.mpl.layers} {bias_str}\n")
        f.write(f"Seed: {self.seed}\n")
        f.write(f"Epochs\tLoss\n")
        results = [str(results[i][0]) + "\t" + str(results[i][1]) + '\n'
                   for i in range(len(results))]
        f.writelines(results)
        f.close