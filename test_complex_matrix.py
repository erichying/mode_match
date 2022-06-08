import numpy as np

a = np.array([[1, 2], [3, 4]])
v = np.array([[6], [9]])
b = np.array([[2, 3], [4, 5]])


print(np.concatenate((a, b), axis=1))
