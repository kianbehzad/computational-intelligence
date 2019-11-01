import perceptron
import numpy as np
import matplotlib.pyplot as plt


pr = perceptron.Perceptron()
input = np.arange(-15, 15, 0.6)
output = []
for i in input:
    output.append(np.sin(i)/i)
output = np.array(output)

pr.train_data(input, output)
pr.make_layer(3)
pr.make_layer(2)
pr.make_layer(1)
pr.eta = 0.01
pr.max_error = 0.0001
pr.execute()
pr.save_weights("sinc")
# pr.load_weights("sinc.list")
# print(pr.get_output(3))
# print(np.sin(3)/3)