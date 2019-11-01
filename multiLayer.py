import numpy as np
from sigmoid import sigmoid, dSigmoid
import matplotlib.pyplot

error = 0
y = np.array([[1, 1, -1], [1, 0, -1], [0, 1, -1], [0, 0, -1]], dtype=float)
d = np.array([0, 1, 1, 0], dtype=float)
w1 = np.array([[.1, .2, .3], [.2, .3, .1]], dtype=float)
w2 = np.array([[.1, .1]], dtype=float)
a = np.array([0, 0], dtype=float)
counter = 0
step = 0

while True:
    a[0] = sigmoid(np.dot(w1[0], np.transpose(y[counter])))
    print("a[0] {}".format(a[0]))
    a[1] = sigmoid(np.dot(w1[1], np.transpose(y[counter])))
    print("a[1] {}".format(a[1]))
    o = np.dot(w2, a)
    print("o {}".format(o))
    error = error + (o - d[counter]) ** 2
    print("error {}".format(error))
    if counter == 3:
        step += 1
        print("error in {} -> {}".format(step, error))
        if error < .001:
            break
        error = 0
        counter = -1
    F = np.array([[dSigmoid(a[0]), 0], [0, dSigmoid(a[1])]])
    print(F)
    s2 = -0.01 * 1 * (d[counter] - o)
    s1 = np.dot(np.dot(F, np.transpose(w2)), s2)
    print("s1 {}".format(s1))
    print("s2 {}".format(s2))

    w1 = w1 - np.dot(np.transpose(np.array([np.dot(np.dot(F, np.transpose(w2)), s2)])), np.array([y[counter]]))
    w2 = w2 - s2 * a
    print("w1 {}".format(w1))
    print("w2 {}".format(w2))

    counter += 1
    break

print("---------------")
for cnt in range(4):
    a[0] = sigmoid(np.dot(w1[0], np.transpose(y[cnt])))
    a[1] = sigmoid(np.dot(w1[1], np.transpose(y[cnt])))
    o = np.dot(w2, a)
    print("{} -> {}".format(y[cnt], o))