import numpy as np
import random
import pickle
import signal
from time import gmtime, strftime
from dateutil import tz
from datetime import datetime
from sigmoid import sigmoid, dSigmoid


class Perceptron():
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self.number_of_layers = 0
        self.number_of_last_layer_norons = 0
        self.eta = 0.01
        self.error = 0
        self.max_error = 0.01
        self.counter = 0
        self.step = 0

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

        # the s value of each layer
        # s[0] is values for first layer
        self.s = []

    def exit_gracefully(self):
        from_zone = tz.tzutc()
        to_zone = tz.tzlocal()
        utc = datetime.strptime('2011-01-21 02:37:21', '%Y-%m-%d %H:%M:%S')
        utc = utc.replace(tzinfo=from_zone)
        central = utc.astimezone(to_zone)
        print(central)
        self.save_weights("weights" + central)

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
        self.s.append(tmp)

    def make_random_weights(self):
        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                for k in range(len(self.w[i][j])):
                    self.w[i][j][k] = random.randint(0, 50) / 100

    def activation_function_diagonal(self, layer_number):
        if layer_number == len(self.w):
            temp = np.ones(len(self.w[-1]))
            return np.diagflat(temp)
        temp = np.zeros(len(self.w[layer_number-1]))
        for i in range(len(self.w[layer_number-1])):
            temp[i] = dSigmoid(self.outputs[layer_number][i])
        return np.diagflat(temp)

    def execute(self):

        while True:
            self.outputs.insert(0, self.y.tolist()[self.counter])
            for i in range(len(self.w)):
                for j in range(len(self.w[i])):
                    weight = np.array(self.w[i][j])
                    input = np.array(self.outputs[i])
                    if i == len(self.w) - 1:
                        self.outputs[i+1][j] = np.dot(weight, input).tolist()
                    else:
                        self.outputs[i+1][j] = sigmoid(np.dot(weight, input).tolist())

            self.error += np.dot(np.transpose(self.outputs[-1] - self.d[self.counter]), (self.outputs[-1] - self.d[self.counter]))

            if self.counter == len(self.y)-1:
                self.step += 1
                print("error in {} -> {}".format(self.step, self.error))
                if self.error < self.max_error:
                    del self.outputs[0]
                    break
                self.error = 0
                self.counter = -1

            self.s[-1] = (1 * self.eta * np.dot(self.activation_function_diagonal(len(self.w)), (self.outputs[-1] - self.d[self.counter]))).tolist()
            for i in range(len(self.w)-2, -1, -1):
                F = self.activation_function_diagonal(i+1)
                self.s[i] = np.dot(np.dot(F, np.transpose(self.w[i+1])), self.s[i+1]).tolist()

            for i in range(len(self.w)-1):
                F = self.activation_function_diagonal(i+1)
                self.w[i] = (self.w[i] - np.dot(np.transpose(np.array([np.dot(np.dot(F, np.transpose(self.w[i+1])), self.s[i+1])])), np.array([self.outputs[i]]))).tolist()
            self.w[-1] = (self.w[-1] - np.multiply(self.s[-1], self.outputs[-2])).tolist()
            self.counter += 1
            del self.outputs[0]

    def save_weights(self, filename):
        with open(filename+'.list', 'wb') as file:
            pickle.dump(self.w, file)
        print(filename + " list saved")

    def load_weights(self, filename):
        self.w = []
        with open(filename, 'rb') as file:
            self.w = pickle.load(file)
        print(filename + " loaded")

    def get_output(self, input):
        input = np.append(input, [-1])
        my_output = np.copy(self.outputs).tolist()
        my_output.insert(0, input)
        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                weight = np.array(self.w[i][j])
                input = np.array(my_output[i])
                if i == len(self.w) - 1:
                    my_output[i + 1][j] = np.dot(weight, input).tolist()
                else:
                    my_output[i + 1][j] = sigmoid(np.dot(weight, input).tolist())
        return my_output[-1]

# pr = Perceptron()
# input = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=float)
# d = np.array([0, 1, 1, 0], dtype=float)
# pr.train_data(input, d)
# pr.make_layer(2)
# pr.make_layer(1)
# # pr.execute()
# # pr.save_weights("xor")
# pr.load_weights("xor.list")
# print("-------------")
# print("{} -> {}".format(input[0], pr.get_output(input[0])))
# print("{} -> {}".format(input[1], pr.get_output(input[1])))
# print("{} -> {}".format(input[2], pr.get_output(input[2])))
# print("{} -> {}".format(input[3], pr.get_output(input[3])))