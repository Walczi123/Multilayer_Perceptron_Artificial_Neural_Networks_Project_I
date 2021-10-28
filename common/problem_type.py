import enum

class problem_type(enum.Enum):
   Classification = 1
   Regression = 2

problem_type.Classification.__name__ = 'classification'
problem_type.Regression.__name__ = 'regression'