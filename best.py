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
store2 = [o for o in store if len(o) > 10]

store, points = edge1.connect1(img.T, edge1.edge)
store = edge1.connect2(store)
store = [[[o2[1], o2[0]] for o2 in o1] for o1 in store]
store2.extend([o for o in store if len(o) > 10])
edge1.plot(store2)
#plt.imshow(img, cmap='gray', vmin=0, vmax=256)
#plt.show()

double_side = store2
one_side = []
no_side = []

for o1 in double_side:
    o1 = np.array(o1)
    s1 = o1[0] - o1[10]
    arg1 = np.arctan2(s1[1], s1[0])
    arg1 = arg1 + np.pi if arg1 < 0 else arg1
    arg1 = arg1 - np.pi if arg1 >= np.pi / 2 else arg1
    arg1 = arg1 * 180 / np.pi
    print(arg1)
    for o2 in double_side:
        pass


















