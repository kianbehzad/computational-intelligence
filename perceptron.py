import numpy as np

def activation_func(x: float):
    if(x > 0):
        return 1
    return 0

w = np.array([0.1, 0.1, 0.5], dtype=float)
y = np.array([[1, 1, -1], [1, 0, -1], [0, 1, -1], [0, 0, -1]])
d = np.array([1, 1, 1, 0])

k = 0
p = 0
e = 0

while True:
    net = np.dot(w, y[p])
    o = activation_func(float(net))
    r = d[p] - o
    e = e + r**2
    deltaW = r * y[p]
    w = w + deltaW * 0.1
    p = p+1
    k = k+1
    if(p >= 3):
        p = 0
        if(e == 0):
            break
        e = 0

print(w)
for i in range(4):
    net = np.dot(w, y[i])
    o = activation_func(float(net))
    print(o)