from prep import get_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
from functions import ZhangSuen

img = get_image(4)
_, img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#_, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

out = ZhangSuen(img)

plt.imshow(out)
plt.show()




















