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

i1 = 0
while i1 < len(store):
    o1 = store[i1]
    for i2 in range(len(store) - 1, -1, -1):
        o2 = store[i2]
        if o1[0][0] <= o2[-1][0]:
            continue
        print(i1, i2)
        break
    i1 += 1


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




















