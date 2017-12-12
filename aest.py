from prep import get_image
import clist
from bed import *
import mathlib as ml
import numpy as np
from scipy import ndimage
import cv2
from skimage.draw import circle_perimeter, circle
import matplotlib.pyplot as plt

img = get_image(0)
_, img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = [np.transpose(o, (1, 0, 2))[0] for o in contours]
contours = np.flip(contours, 2)
contours = contours[0]
len_c = lambda i: i % len(contours)

img_c = np.zeros((28, 28))
for c in contours:
    img_c[c[0], c[1]] = 1

def init(key_i):
    directions = [[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]]
    sa = contours[len_c(key_i + 1)] - contours[len_c(key_i - 1)]
    d = np.array(ml.quantize_dir([-sa[1], sa[0]]))
    p = contours[key_i] + d
    while img_c[p[0], p[1]] == 0 or img[p[0], p[1]] == 0:
        p += d
    for i, c in enumerate(contours):
        if np.all(c == p):
            return i
def line_angle(i1, i2):
    p1 = clist.get(contours, i1 - 1, i1 + 2)
    p2 = clist.get(contours, i2 - 1, i2 + 2)
    m1, m2 = (p1[0] + p1[2]) / 2.0, (p2[0] + p2[2]) / 2.0
    ea1 = ml.exterior_angle([p1[0], m1, m2])
    ea2 = ml.exterior_angle([p2[0], m2, m1])
    return [ea1, ea2]
facing_gap = lambda ea: abs(ea[0] + ea[1] + 180)
facing_agl = lambda ea: abs(ea[0] - ea[1])
def init2(i1, i2):
    while True:
        c = facing_gap(line_angle(i1, i2))
        p = facing_gap(line_angle(i1, i2 + 1))
        m = facing_gap(line_angle(i1, i2 - 1))
        if c <= p and c <= m:
            return i2
        elif p < m:
            i2 += 1
        else:
            i2 -= 1
def move(i1, i2, s):
    pp = np.array([[1, 0], [0, -1], [1, -1]])
    pp *= s
    l = [[i, facing_gap(line_angle(i1 + p[0], i2 + p[1]))] for i, p in enumerate(pp)]
    pi = sorted(l, key=lambda t: t[1])[0][0]
    return [i1 + pp[pi][0], i2 + pp[pi][1]]
remainings = [True for e in range(len(contours))]
i1 = 0
i2 = init(i1)
i2 = init2(i1, i2)

for n in range(5):
    fa = facing_agl(line_angle(i1, i2))
    print(fa)
    draw = lambda p1, p2: plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r')
    p1, p2 = contours[i1], contours[i2]
    #draw(p1, p2)
    if n >= 4:
        draw(contours[i1 - 1], contours[i1 + 1])
        draw(contours[i2 - 1], contours[i2 + 1])
    i1, i2 = move(i1, i2, -1)

#tmp = np.transpose(contours, (1, 0))
#plt.plot(tmp[1], tmp[0], 'ro')
plt.imshow(img)
plt.show()




















