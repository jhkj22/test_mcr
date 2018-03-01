import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools

img = cv2.imread('res/kick_boxing.jpg', 0)
#img = img[10: 70, 280: 340]
def remove_edge(contours):
    start, stop = -1, -1
    for n, p in enumerate(contours):
        f = np.any((p == 0) | (p == 9))
        if f:
            if start != -1:
                stop = n
                break
        else:
            if start == -1:
                start = n
    if start == 0:
        cont = contours[: stop + 1]
        return np.concatenate((contours[-2: -1], cont))
    elif stop == -1:
        cont = contours[start - 1:]
        return np.concatenate((cont, contours[0: 1]))
    return contours[start - 1: stop + 1]
def detect_edge(img):
    ret,th = cv2.threshold(img,0, 1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(th,
                                cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        return np.array([])
    th = np.where(th == 1, 0, 1).astype(np.uint8)
    _, contours_r, _ = cv2.findContours(th,
                                cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        return np.array([])
    contours = contours[0].reshape(-1, 2)
    contours = remove_edge(contours)
    contours = np.flip(contours, axis=1)
    return contours

def plot_edge(edge, r, c):
    edge = np.transpose(edge)
    plt.plot(edge[1] + c, edge[0] + r, 'ro', markersize=2)
def is_bactrian(img):
    b = np.histogram(img.flatten())
    d = b[1][-1] - b[1][0]
    if d < 20:
        return False
    b = b[0]
    for sep in range(len(b) - 1):
        sum_mn = min([np.sum(b[: sep + 1]), np.sum(b[sep + 1:])])
        sum_r = sum_mn / 100
        num_mx = max(b[sep: sep + 2])
        num_r = num_mx / 100
        #print(sum_r, num_r)
        if sum_r > 0.1 and num_r < 0.05:
            return True
    return False

def plot_rect(r, c):
    plt.plot([c, c + 10, c + 10, c, c],
             [r, r, r + 10, r + 10, r], 'r')
def abc(out, r, c):
    clip = img[r: r + 10, c: c + 10]
    if not is_bactrian(clip):
        return -1
    edge = detect_edge(clip)
    if len(edge) == 0:
        return -1
    plot_edge(edge, r, c)
    #edge = edge + np.array([r, c])
    #for ed in edge:
    #    out[ed[0], ed[1]] = 1

out = np.zeros(img.shape)
for r, c in itertools.product(range(0, img.shape[0] - 5, 5),
                              range(0, img.shape[1] - 5, 5)):
    abc(out, r, c)
"""
for r, c in itertools.product(range(img.shape[0]), range(img.shape[1])):
    if out[r, c] == 0:
        continue
    print(r, c)
    break
"""

#out = cv2.Canny(img, 100, 300)
#ps = np.where(out == 255)
#plt.plot(ps[1], ps[0], 'ro', markersize=2)
plt.imshow(img, cmap='gray', vmin=0, vmax=256)
plt.show()
















