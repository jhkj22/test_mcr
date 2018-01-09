import numpy as np
import matplotlib.pyplot as plt
import cv2
from bed import *

img = cv2.imread('res/apple.jpg', 0)
img = img[120: 170, 90: 140]

store = []
tmp = []
points = []

for r in range(img.shape[0]):
    line = img[r].astype(np.int)
    diff = np.abs(line[:-1] - line[1:])
    l = []
    for c in range(1, len(diff) - 1):
        if diff[c] < 20:
            continue
        if diff[c] > diff[c - 1] and diff[c] > diff[c + 1]:
            l.append(c)
            points.append([r, c])
    i, j = 0, 0
    while i < len(tmp) and j < len(l):
        if np.abs(tmp[i][-1][1] - l[j]) <= 1:
            tmp[i].append([r, l[j]])
            l.pop(j)
            i += 1
        elif tmp[i][-1][1] < l[j]:
            i += 1
        else:
            j += 1
    i = len(tmp) - 1
    while i >= 0:
        if tmp[i][-1][0] != r:
            store.append(tmp[i])
            tmp.pop(i)
        i -= 1
    for o in l:
        tmp.append([[r, o]])
    tmp = sorted(tmp, key=lambda t: t[-1][1])
for o in tmp:
    store.append(o)
store_h = sorted(store, key=lambda t: t[0][0])

for n1, o1 in enumerate(store):
    if n1 == 0:
        print(o1)
        print('')
    for n2, o2 in enumerate(store_h):
        if o1[-1][0] >= o2[0][0]:
            continue
        if len(o1) < 3 and len(o2) < 3:
            continue
        if n1 == 0:
            print(o2)

#for o in store_h:
#    print(o)
"""
for ps in store:
    if len(ps) < 5:
        continue
    ps = np.transpose(ps)
    plt.plot(ps[1], ps[0], 'r')

points = np.transpose(points)
plt.plot(points[1], points[0], 'bo', markersize=1)
plt.imshow(img, cmap='gray', vmin=0, vmax=256)

plt.show()
"""




















