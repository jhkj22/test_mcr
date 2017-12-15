import numpy as np
import matplotlib.pyplot as plt
import cv2
from bed import *

img = cv2.imread('res/cat.jpg', 0)

img = img[340: 410, 230: 300]
img = img >> 5
out = np.where(img == 0, 1, 0)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(out, cmap='gray')
plt.show()
























