import numpy as np
import matplotlib.pyplot as plt
import cv2
from bed import *
import edge1
import shadow_edge

img = cv2.imread('res/cat.jpg', 0)
#img = img[10: 70, 280: 340]
#img = img[200: 300, 100: 200]
img = img[350: 400, 300: 350]

"""
plt.subplot(121)
store1, points1 = edge1.connect1(img, edge1.edge)
l = np.transpose(points1)
plt.plot(l[1], l[0], 'ro', markersize=2)
store1 = edge1.connect2(store1)
store1 = [o for o in store1 if len(o) > 10]
"""
"""
store2, points2 = edge1.connect1(img.T, edge1.edge)
store2 = edge1.connect2(store2)
store2 = [[[o2[1], o2[0]] for o2 in o1] for o1 in store2]
l = np.transpose(points2)
plt.plot(l[0], l[1], 'bo', markersize=2)
#store2 = [o for o in store2 if len(o) > 10]
"""
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
for n in range(24, 27):
    line = img[n].astype(np.int)
    dif = line[1:] - line[:-1]
    plt.plot(line)
plt.show()
















