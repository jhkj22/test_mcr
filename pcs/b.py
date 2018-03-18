import numpy as np
import matplotlib.pyplot as plt
import random

img = np.zeros((500, 500))

for i in range(300):
    x = random.random() * 500
    y = random.random() * 500
    x, y = int(x), int(y)
    img[x, y] = 1
for i in range(3000):
    x = random.random() * 500
    y = random.random() * 500
    x, y = int(x), int(y)
    if np.sqrt((x - 250) ** 2 + (y - 250) ** 2) <= 150:
        img[x, y] = 1

plt.imshow(img)
plt.show()





