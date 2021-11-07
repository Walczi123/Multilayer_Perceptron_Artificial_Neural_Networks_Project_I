import numpy as np

a = [-1.06630768, -0.14151592, -0.06925467, -0.17085654,  1.68768015, -0.00863609, -0.36610732, -0.27355028,  0.44715848, -1.05332099]
a = np.array(a)

exps = np.exp(a)
result = exps / exps.sum()
print(result)