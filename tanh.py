import math
import numpy as np

def tanh(x: float):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def tanhmatrix(a: np.ndarray):
    sumTanh = 0
    for i in np.nditer(a):
        sumTanh += tanh(i)
    return sumTanh

a = np.array([[1, 2, 3], [1, 2, 3]])
print(tanhmatrix(a))