import numpy as np
import matplotlib.pyplot as plt
import cv2


def distance(center, input):
    sum = 0
    for i in range(len(input)):
        sum += (center[i]-input[i])**2
    return np.sqrt(sum)


def make_center(input: np.ndarray):
    sum = np.zeros((1, len(input[0])))
    for i in range(len(input)):
        sum += input[i]
    return sum/len(input)


def make_mask(data, centers):
    mask = np.zeros((len(data), 1))
    for i in range(len(data)):
        min_dist = 10000000
        index = -1
        for j in range(len(centers)):
            if distance(centers[j], data[i]) < min_dist:
                min_dist = distance(centers[j], data[i])
                index = j
        mask[i] = index
    return mask

def k_Mean(data, k):
    centers = np.zeros((k, len(data[0])))
    last_centers = np.zeros((k, len(data[0])))

    for i in range(k):
        centers[i] = np.random.randint(0, 50, len(data[0]))


    while True:

        mask = make_mask(data, centers)

        for i in range(k):
            my_list = []
            for j in range(len(mask)):
                if mask[j] == i:
                    my_list.append(data[j])
            #print(np.array(my_list))
            if len(my_list) == 0:
                return k_Mean(data, k)
            centers[i] = make_center(np.array(my_list))
        error = 0
        for i in range(len(centers)):
            error += distance(centers[i], last_centers[i])
        if error < 1:
            return mask, centers
        last_centers = centers




# lena = cv2.imread('lena.jpg', 0)
# lena = np.reshape(lena, (1, -1))
# lena = np.transpose(lena)
#
# k = 5
# mask, centers = k_Mean(lena,k)
# for i in range(len(lena)):
#   lena[i] = centers[int(mask[i])]
#
# lena = np.transpose(lena)
# lena = np.reshape(lena, (446, 651))
# cv2.imwrite("mylena.png", lena)
# cv2.imshow(lena)

data = np.array([[1, 2], [3, 4], [1, 5], [10, 10], [10, 50], [100, 20], [20, 25], [10, 50]])
k = 4
mask, centers = k_Mean(data, k)
fig, ax = plt.subplots()
template = ['ro', 'bo', 'go', 'yo', 'r*', 'b*', 'g*', 'y*']
for i in range(len(data)):
    ax.plot(data[i][1], data[i][0], template[int(mask[i])])
plt.show()


