import math
import numpy as np

def sigmoid(x: float):
    return (2 / (1 + math.exp(-x))) - 1

def dSigmoid(x: float):
    #return (2*math.exp(-x)) / ((1 + math.exp(-x)) ** 2)
    return (1 - sigmoid(x)**2) / 2

def sigmatrix(a: np.ndarray):
    sumSigmoid = 0
    sumDSigmoid = 0
    for i in np.nditer(a):
        sumSigmoid += sigmoid(i)
        sumDSigmoid += dSigmoid(i)
    return (sumSigmoid, sumDSigmoid)

a = np.array([[1, 2, 3], [1, 2, 3]])
print(sigmatrix(a))