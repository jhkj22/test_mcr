import numpy as np
import matplotlib.pyplot as plt
import cv2
from bed import *
import edge1
import shadow_edge

img = cv2.imread('res/banana.jpg', 0)
#img = img[10: 70, 280: 340]
def a(img, r, c):
    ret,th = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(th,
                                cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 1:
        contours = np.transpose(contours[0].reshape(-1, 2))
        plt.plot(contours[0] + c, contours[1] + r, 'ro')
        return ret
    return -1
    
def is_bactrian(img):
    b = np.histogram(img.flatten())
    d = b[1][-1] - b[1][0]
    if d < 4:
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
def screen_out(img):
    mx, mn = np.max(img), np.min(img)
    mx, mn = float(mx), float(mn)
    d = mx - mn
    n = d // 12
    for i in range(1, 13):
        th_v = mn + i * n
        _, th = cv2.threshold(img, th_v, 1, cv2.THRESH_BINARY)
        plt.subplot(3, 4, i)
        plt.imshow(th, cmap='gray')
        plt.text(0, 0, str(th_v), color='r')

for r in range(0, img.shape[0], 10):
    for c in range(0, img.shape[1], 10):
        clip = img[r: r + 10, c: c + 10]
        if is_bactrian(clip):
            result = a(clip, r, c)

plt.imshow(img, cmap='gray', vmin=0, vmax=256)
plt.grid()
plt.xticks(np.arange(-0.5, img.shape[1], 10))
plt.yticks(np.arange(-0.5, img.shape[0], 10))

plt.show()
















