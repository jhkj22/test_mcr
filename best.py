import numpy as np
import matplotlib.pyplot as plt
import cv2
from bed import *

img = cv2.imread('res/cat.jpg', 0)

img = img[340: 410, 230: 300]
"""
l = []
for c in range(70):
    for r in range(69):
        if img[r, c] != img[r + 1, c]:
            l.append([r + 0.5, c])
l = np.transpose(l)
plt.plot(l[1], l[0], 'r_')

l = []
for r in range(70):
    for c in range(69):
        if img[r, c] != img[r, c + 1]:
            l.append([r, c + 0.5])
l = np.transpose(l)
plt.plot(l[1], l[0], 'r|')
"""
img = img >> 5
line = img[30]
plt.subplot(211)
plt.imshow(img, cmap='gray')
plt.subplot(212)
plt.plot(range(len(line)), line)
plt.show()
























