import clist
import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
import cv2
import random
from scipy import signal
"""
z, p, k = signal.ellip(12, 1, 100, 1 / 11, output='zpk')
for n in range(6):
    print('{{'+str(z[n])[1:-1]+'},{'+str(z[n+6])[1:-1]+'}}')

fig, ax = plt.subplots()
ax.axis('equal')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.add_artist(plt.Circle((0, 0), 1, fill=False))
ax.scatter(z.real, z.imag)
ax.scatter(p.real, p.imag, marker='x')
#w, h = signal.freqz(b, a)
#plt.plot(w / np.pi, 20 * np.log10(abs(h)))
plt.show()
"""

x = np.array([[0] * 200 + list(np.arange(0, 1, 0.01) ** 2 * 256) + [255] * 200] * 200)
plt.imshow(x, cmap='gray')
l = [0] * 200 + list(np.arange(0, 1, 0.01) ** 2 * 256) + [255] * 200
#plt.plot(range(len(l)), l)
plt.show()

















