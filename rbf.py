import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

noron_num = 4

input = np.linspace(-3, 3, 100)
outputDesired = np.sin(input)
outputNoisy = outputDesired + np.random.normal(0, 0.1, input.shape)
# plt.plot(input,outputDesired)
# plt.plot(input,outputNoisy)
# plt.show()

kmeans = KMeans(n_clusters= noron_num,random_state=0).fit(input.reshape(-1, 1))
centers = kmeans.cluster_centers_

max_dist = -1
for i in range(noron_num):
    for j in range(noron_num):
        dist = abs(centers[i] - centers[j])
        if dist > max_dist:
            max_dist = dist

matrix = np.ones((noron_num+1, len(input)))
for i in range(noron_num):
    matrix[i+1] = gaussian(input, centers[i], noron_num / (np.sqrt(2 * max_dist)))
matrix = np.transpose(matrix)
W = np.dot(np.linalg.inv(np.dot(np.transpose(matrix),matrix)),np.transpose(matrix))
W = np.dot(W,outputNoisy)

actual_output = []
for i in range(len(input)):
    actual_output.append(np.dot(matrix[i], W))
plt.plot(input, actual_output, 'b')
plt.plot(input, outputNoisy, 'r')
plt.plot(input, outputDesired, 'black', alpha=0.7)
plt.show()