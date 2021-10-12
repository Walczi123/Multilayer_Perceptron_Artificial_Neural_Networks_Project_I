import pandas as pd
from common.problem_type import problem_type


class Reader:
    """
    Class which reads the data files and prepares a proper configuration of the neural network.
    """

    def read_file(self, filename: str, ):
        try:
            return pd.read_csv(filename).values.tolist()
        except:
            raise FileNotFoundError('Could not find the file')
    
    def prepare_data(self, p_type: problem_type, filename: str):
        try:
            data = self.read_file(filename)

            if p_type == problem_type.Regression:
                return data
            elif p_type == problem_type.Classification:
                return [[[row[0], row[1]], row[2]] for row in data]
        except:
            raise
        
        raise ValueError("The problem type was wrong")


if __name__ == "__main__":
    reader = Reader()
    try:
        # result = reader.prepare_data(problem_type.Regression,  "data/regression/data.activation.test.100.csv")
        result = reader.prepare_data(problem_type.Classification,  "data/classification/data.simple.test.100.csv")
    except Exception as e:
        print(e)
        exit()

    print(result)
    print(len(result))
