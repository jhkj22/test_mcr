import clist
import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
import cv2
import random
from scipy import signal

A = np.random.randint(0, 10, 10000)
A = np.where(A < 5, 0, 1).reshape((100, 100))
plt.imshow(A, cmap='gray')
plt.show()


















