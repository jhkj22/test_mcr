import numpy as np
import matplotlib.pyplot as plt
import cv2
from bed import *
import edge1
import shadow_edge

img = cv2.imread('res/banana.jpg', 0)
img = img[10: 70, 280: 340]
#img = img[25: 35, 20: 30]
img = img[40: 50, 5: 15]
#img = img[200: 300, 100: 200]
#img = img[350: 400, 300: 350]

_, th = cv2.threshold(img, 180, 1, cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(np.array(th), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = contours[0]
contours = contours.reshape(-1, 2)
"""
for n in range(12):
    plt.subplot(3, 4, n + 1)
    th = 180 + n * 5
    out = np.where(img > th, 1, 0)
    plt.imshow(out, cmap='gray')
    plt.text(0, 0, str(th), color='r')
"""
c = np.transpose(contours)
plt.plot(c[0], c[1], 'ro')
plt.imshow(th, cmap='gray')
plt.show()
















