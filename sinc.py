from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt


pr = Perceptron()
input = np.arange(-15, 15, 0.7)
output = []
for i in input:
    output.append(np.sin(i))
output = np.array(output)

pr.train_data(input, output)
pr.make_layer(3)
pr.make_layer(2)
pr.make_layer(3)
pr.make_layer(4)
pr.make_layer(2)
pr.make_layer(1)
pr.eta = 1
pr.max_error = 5
pr.execute()
pr.save_weights("sinc")
# pr.load_weights("sinc.list")
# print(pr.get_output(3))
# print(np.sin(3)/3)