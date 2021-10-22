import numpy as np
import pandas as pd
from common.problem_type import problem_type


def normalize_data(dataset):
    norm = np.linalg.norm(dataset)
    return dataset/norm


def read_file(filename: str, ):
    try:
        return pd.read_csv(filename).values.tolist()
    except:
        raise FileNotFoundError('Could not find the file')


def prepare_data(p_type: problem_type, filename: str):
    try:
        data = normalize_data(read_file(filename))
        if p_type == problem_type.Regression:
            dataset = []
            for row in data:
                dataset.append([np.array([row[0]]), row[1]])
            return dataset
        elif p_type == problem_type.Classification:
            dataset = []
            for row in data:
                dataset.append([np.array([row[0], row[1]]), row[2]])
            return dataset
    except:
        raise

    raise ValueError("The problem type was wrong")


if __name__ == "__main__":
    try:
        result = prepare_data(problem_type.Regression,
                              "data/regression/data.activation.test.100.csv")
        # result = prepare_data(problem_type.Classification,  "data/classification/data.simple.test.100.csv")
    except Exception as e:
        print(e)
        exit()

    print(result)
    print(normalize_data(result))
    print(len(result))
