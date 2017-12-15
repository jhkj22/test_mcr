import clist
import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
import cv2
import random
from scipy import signal

A = np.zeros((50, 50))

def neighbor(p):
    l = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    l = list(l + np.array(p))
    l = [o for o in l if A[o[0], o[1]] == 0]
    return l
l = [[25, 25]]
for _ in range(1000):
    n = random.randint(0, len(l) - 1)
    p = l.pop(n)
    A[p[0], p[1]] = 1
    l.extend(neighbor(p))
    for o in l:
        A[o[0], o[1]] = 2
A = np.where(A == 1, 1, 0)
plt.imshow(A, cmap='gray')
plt.show()
















