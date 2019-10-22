import math
import numpy as np

# def sigmoid(x: float):
#     return (1 / (1 + math.exp(-x)))

def sigmoid(x: np.ndarray):
    if type(x) == np.ndarray:
        a = np.arange(x.shape[0] * x.shape[1], dtype=np.float).reshape(x.shape[0], x.shape[1])
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i][j] = sigmoid(x[i][j])
        return a

    #if type(x) == np.int64 or type(x) == np.float or type(x) == int or type(x) == float:
    else:
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

# a = np.array([[1, 2, 3], [4, 5, 10]])
# print(sigmoid(a))
