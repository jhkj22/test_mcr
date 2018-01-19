import numpy as np
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

def distance(o1, o2):
    a = o1[0] - o2[0]
    b = o1[1] - o2[1]
    return (a ** 2 + b ** 2) ** 0.5

def joint(o1, o2):
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
        if abs(i1) > 30 or i2 > 30:
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
    debug = False
    if len(tmp) == 217:
        debug = True
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

"""
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
"""
i = 5
for o1 in store_new[i: i + 1]:
    l = np.transpose(o1)
    plt.plot(l[1], l[0], 'r')
plt.imshow(img, cmap='gray')
plt.show()
"""















