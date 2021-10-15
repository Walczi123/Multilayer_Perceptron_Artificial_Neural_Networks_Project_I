import pandas as pd
from common.problem_type import problem_type


def read_file( filename: str, ):
    try:
        return pd.read_csv(filename).values.tolist()
    except:
        raise FileNotFoundError('Could not find the file')

def prepare_data(p_type: problem_type, filename: str):
    try:
        data = read_file(filename)

        if p_type == problem_type.Regression:
            return data
        elif p_type == problem_type.Classification:
            return [[[row[0], row[1]], row[2]] for row in data]
    except:
        raise
    
    raise ValueError("The problem type was wrong")


if __name__ == "__main__":
    try:
        # result = prepare_data(problem_type.Regression,  "data/regression/data.activation.test.100.csv")
        result = prepare_data(problem_type.Classification,  "data/classification/data.simple.test.100.csv")
    except Exception as e:
        print(e)
        exit()

    print(result)
    print(len(result))
