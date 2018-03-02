from prep import get_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools

img = get_image(0)
_, img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#_, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)




def scan(ps):
    center = []
    count = 0
    for i, p in enumerate(ps):
        if img[p[0], p[1]] == 1:
            count += 1
        else:
            if count > 0:
                center.append([i - (count // 2 + 1), count])
                count = 0
    return center
def thickness_1d(edge, d):
    comp = []; uncomp = []
    for s in edge:
        line = []
        while np.all((0 <= s) & (s < img.shape[1])):
            line.append(np.copy(s))
            s += d
        head = [[line[i], s] for i, s in scan(line)]
        j = len(uncomp) - 1
        while j >= 0:
            i = len(head) - 1
            cont = False
            while i >= 0:
                if np.max(np.abs(uncomp[j][-1][0] - head[i][0])) < 3 and abs(uncomp[j][-1][1] - head[i][1]) < 4:
                    uncomp[j].append(head[i])
                    head.pop(i)
                    cont = True
                i -= 1
            if not cont:
                comp.append(uncomp[j])
                uncomp.pop(j)
            j -= 1
        for h in head:
            uncomp.append([h])
    comp = [c for c in comp if len(c) >= min(c, key=lambda t: t[1])[1]]
    return comp

def thickness():
    start = []
    for r in range(img.shape[0]):
        start.append([r, 0])
    return thickness_1d(np.array(start), np.array([0, 1]))

def plot_comp(comp):
    l = []
    for c in comp:
        x, y = [], []
        for p in c:
            x.append(p[0][1]); y.append(p[0][0])
        l.append([x, y])
    return l
comp = thickness()
comp = plot_comp(comp)
for c in comp:
    plt.plot(c[0], c[1], 'r')

img = img.T
comp = thickness()
comp = plot_comp(comp)
for c in comp:
    plt.plot(c[1], c[0], 'b')
plt.imshow(img.T)
plt.show()




















