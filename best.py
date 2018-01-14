import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2
from bed import *
import edge1
import shadow_edge

img = cv2.imread('res/apple.jpg', 0)
img = img#[290: 390, 170: 270]

store, points = edge1.connect1(img, edge1.edge)
store = edge1.connect2(store)
edge1.plot(store)

store, points = edge1.connect1(img.T, edge1.edge)
store = edge1.connect2(store)
edge1.plot(store, reverse=True, color='b')


plt.imshow(img, cmap='gray', vmin=0, vmax=256)
plt.show()





















