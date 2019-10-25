import numpy as np
import random


class Perceptron():
    def __init__(self):
        self.number_of_layers = 0
        self.number_of_last_layer_norons = 0
        # augmented inputs
        self.y = np.ndarray
        # desired outputs
        self.d = np.ndarray
        # all weights
        # w[0] all wights for layer one
        # w[1][1] inputs weights in second noron from second layer
        self.w = []
        # outputs of each noron
        # output[0] norons of first layer
        # output[1][2] output of third noron in second layer
        self.outputs = []

    def train_data(self, input, output):
        if input.ndim == 1:
            input = np.array([input])
            input = np.transpose(input)
        self.y = np.zeros((input.shape[0], input.shape[1] + 1))
        for i in range(len(input)):
            self.y[i] = np.append(input[i], [-1])
        self.number_of_last_layer_norons = len(self.y[0])

        self.d = output

    def make_layer(self, number_of_norons):
        self.number_of_layers += 1
        temp = np.zeros((number_of_norons, self.number_of_last_layer_norons)).tolist()
        self.number_of_last_layer_norons = number_of_norons
        self.w.append(temp)
        self.make_random_weights()
        tmp = np.zeros(number_of_norons).tolist()
        self.outputs.append(tmp)


    def make_random_weights(self):
        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                for k in range(len(self.w[i][j])):
                    self.w[i][j][k] = random.randint(0, 50) / 100

    def execute(self):
        pass


pr = Perceptron()
input = np.array([[1, -1], [2, -1], [3, -1], [4, -1]], dtype=float)
d = np.array([1, 5, 6, 7], dtype=float)
pr.train_data(input, d)
pr.make_layer(2)
pr.make_layer(1)
