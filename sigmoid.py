import math
import numpy as np

def sigmoid(x: float):
    return (1 / (1 + math.exp(-x)))

def dSigmoid(x: float):
    #return (2*math.exp(-x)) / ((1 + math.exp(-x)) ** 2)
    return sigmoid(x) * (1 - sigmoid(x))

def sigmatrix(a: np.ndarray):
    sumSigmoid = 0
    sumDSigmoid = 0
    for i in np.nditer(a):
        sumSigmoid += sigmoid(i)
        sumDSigmoid += dSigmoid(i)
    return (sumSigmoid, sumDSigmoid)

a = np.array([[1, 2, 3], [1, 2, 3]])
print(sigmatrix(a))