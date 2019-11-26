import numpy as np
import cv2
import matplotlib.pyplot as plt


def sign(a: np.ndarray):
    w = np.sign(a)
    w = w.astype(float)
    for i in range(len(w)):
        if w[i] == 0:
            w[i] = -1
    return w


def asyncron(v_new, v_old):
    a = np.copy(v_old)
    for i in range(len(v_old)):
        if a[i] != v_new[i]:
            a[i] = v_new[i]
            break
    return a


def image_info_extrackter(name):
    l1 = cv2.imread(name)
    l1 = cv2.resize(l1, (10, 10))
    l1 = cv2.cvtColor(l1, cv2.COLOR_BGR2GRAY)
    _, l1 = cv2.threshold(l1, 10, 255, 0)
    l1 = sign(np.reshape(l1, (1, -1))[0])
    return l1


def noisy(name, percent=10):
  out = image_info_extrackter(name)
  l = np.random.randint(100, size=percent).tolist()
  for i in l:
    if out[i] == -1:
      out[i] = 1
    else:
      out[i] = -1
  return out

def debug_print(a: np.ndarray, b: np.ndarray):
    for i in range(len(a)):
        print(a[i], b[i])

def show_image_from_info(input: np.ndarray):
    plt.imshow(input.reshape((10, 10)))
    plt.show()


photo_b = image_info_extrackter("photo_b.jpg")
photo_d = image_info_extrackter("photo_d.jpg")
photo_g = image_info_extrackter("photo_g.jpg")
photo_k = image_info_extrackter("photo_k.jpg")
photo_p = image_info_extrackter("photo_p.jpg")

noisy_b = noisy("photo_b.jpg", 5)
noisy_d = noisy("photo_d.jpg", 0)
noisy_g = noisy("photo_g.jpg", 5)
noisy_k = noisy("photo_k.jpg", 5)
noisy_p = noisy("photo_p.jpg", 5)


pattern = np.transpose(np.concatenate(([photo_b], [photo_d], [photo_g], [photo_k], [photo_p]), axis=0))
m = pattern.shape[0]
n = pattern.shape[1]
W = np.dot(pattern, np.transpose(pattern)) - n * np.eye(m)

# W_temp = np.zeros((100, 100))
# for i in range(5):
#     aa = np.transpose([np.transpose(pattern)[i]])
#     aa = np.dot(aa, np.transpose(aa))
#     W_temp += aa
# W_temp -= n * np.eye(m);
# print(np.allclose(W, W_temp))

input = photo_b
show_image_from_info(input)
v0 = np.transpose(np.array([input]))

v_old = v0
counter = 0
v_new = np.ones((len(v_old), 1))

while True:
    v_new_temp = sign(np.dot(W, v_old))
    debug_print(v_old, v_new_temp)
    break
    counter += 1
    show_image_from_info(v_new)
    print("iter {} -> {}".format(counter, np.transpose(v_new)))
    if np.allclose(v_old, v_new):
        print("result -> {}".format(np.transpose(v_new)))
        show_image_from_info(v_new)
        break
    v_old = v_new

