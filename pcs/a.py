import matplotlib.pyplot as plt
import numpy as np
import copy
from data import getSize, getClose
from extreme import get_ps, get_size
from smooth import gauss

start = getSize() - 10000 + 200
close = np.array(getClose(start, start + 2 ** 7))

close = gauss(close, 8)
ps_u = []
ps_d = []
for i in range(1, len(close) - 1):
    if 2 * close[i] - close[i - 1] > close[i + 1]:
        ps_u.append(i)
    else:
        ps_d.append(i)

plt.plot(close)
plt.plot(ps_u, close[ps_u], 'ro')
plt.plot(ps_d, close[ps_d], 'bo')
plt.show()










