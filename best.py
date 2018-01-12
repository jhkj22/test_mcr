import numpy as np
import matplotlib.pyplot as plt
import cv2
from bed import *
from mathlib import exterior_angle

img = cv2.imread('res/apple.jpg', 0)
img = img#[290: 390, 170: 270]

"""
plt.subplot(122)
for c in [45, 50, 55]:
    plt.plot(img.T[c])
plt.subplot(121)
"""
plt.imshow(img, cmap='gray', vmin=0, vmax=256)
plt.show()





















