import numpy as np
import matplotlib.pyplot as plt
import cv2
from bed import *
import edge1
import shadow_edge

img = cv2.imread('res/banana.jpg', 0)
img = img#[290: 390, 170: 270]


store1, points1 = edge1.connect1(img, edge1.edge)
l = np.transpose(points1)
plt.plot(l[1], l[0], 'ro', markersize=2)
#store1 = edge1.connect2(store1)
#store1 = [o for o in store1 if len(o) > 10]

store2, points2 = edge1.connect1(img.T, edge1.edge)
#store2 = edge1.connect2(store2)
#store2 = [[[o2[1], o2[0]] for o2 in o1] for o1 in store2]
l = np.transpose(points2)
plt.plot(l[0], l[1], 'bo', markersize=2)
#store2 = [o for o in store2 if len(o) > 10]
"""
def distance(o1, o2):
    a = o1[0] - o2[0]
    b = o1[1] - o2[1]
    return (a ** 2 + b ** 2) ** 0.5
def head_vec(head, body):
    s = np.array(head) - np.array(body)
    arg = np.angle(np.complex(s[0], s[1]))
    arg = arg * 180 / np.pi
    return arg
def to_1quadrant(arg):
    arg = arg + 180 if arg < 0 else arg
    arg = arg - 90 if arg >= 90 else arg
    return arg
def joint(o1, o2):
    v1 = head_vec(o1[-1], o1[-11])
    v2 = head_vec(o2[0], o2[10])
    v11 = abs(to_1quadrant(v1) - 45)
    v12 = abs(to_1quadrant(v2) - 45)
    if v11 < 5 and v12 < 5 and abs(abs(v1 - v2) - 180) < 10:
        stop1, stop2 = 40, 40
    else:
        stop1, stop2 = 5, 5
    i1, i2 = -1, 0
    while abs(i1) < len(o1) and i2 < len(o2) - 1:
        d0 = distance(o1[i1], o2[i2])
        d1 = distance(o1[i1 - 1], o2[i2])
        d2 = distance(o1[i1], o2[i2 + 1])
        if d1 <= d0:
            i1 -= 1
        if d2 <= d0:
            i2 += 1
        if d0 < d1 and d0 < d2:
            break
        if abs(i1) > stop1 or i2 > stop2:
            return []
    if d0 < 5:
        return [i1, i2]
    else:
        return []

store_new = []
store = []
store.extend(store1); store.extend(store2)

while len(store) > 1:
    tmp = store.pop(0)
    for e1 in range(2):
        loop_flag = True
        while loop_flag:
            loop_flag = False
            for i1, o1 in enumerate(store):
                for e2 in range(2):
                    l = joint(o1, tmp)
                    if len(l) != 0:
                        if l[0] < -1:
                            o1 = o1[:l[0]]
                        o1.extend(tmp[l[1]:])
                        tmp = o1
                        store.pop(i1)
                        loop_flag = True
                        break
                    o1.reverse()
                if loop_flag:
                    break
        tmp.reverse()
    store_new.append(tmp)
if len(store) > 0:
    store_new.append(store.pop(0))


for o1 in store_new:
    l = np.transpose(o1)
    plt.plot(l[1], l[0], 'r')
"""
plt.imshow(img, cmap='gray')
plt.show()
















