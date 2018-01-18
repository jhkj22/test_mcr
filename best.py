import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2
from bed import *
import edge1
import shadow_edge

img = cv2.imread('res/apple.jpg', 0)
img = img#[290: 390, 170: 270]


store1, points = edge1.connect1(img, edge1.edge)
store1 = edge1.connect2(store1)
store1 = [o for o in store1 if len(o) > 10]

store2, points = edge1.connect1(img.T, edge1.edge)
store2 = edge1.connect2(store2)
store2 = [[[o2[1], o2[0]] for o2 in o1] for o1 in store2]
store2 = [o for o in store2 if len(o) > 10]

def head_vec(head, body):
    s = head - body
    arg = np.angle(np.complex(s[0], s[1]))
    arg = arg * 180 / np.pi
    return arg

for o in store1:
    o = np.array(o)
    arg = head_vec(o[0], o[10])
    s = 0
    if abs(arg - 135) < 5:
        s += 1
    elif abs(arg + 135) < 5:
        s += 2
    arg = head_vec(o[-1], o[-11])
    if abs(arg - 45) < 5:
        s += 10
    elif abs(arg + 45) < 5:
        s += 20
    print(s)

"""
l = np.transpose(store2[5])
plt.plot(l[1], l[0], 'r')
plt.imshow(img, cmap='gray')
plt.show()
"""















