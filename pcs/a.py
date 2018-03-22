import matplotlib.pyplot as plt
import numpy as np
import copy
from data import getSize, getClose
from extreme import get_ps, get_size, draw_ps
from smooth import gauss

start = getSize() - 10000 + 300
close = np.array(getClose(start, start + 2 ** 7))

ps = get_ps(close)
psu = ps[0::2]
psd = ps[1::2]
ps = np.transpose(psu)
plt.plot(ps[0], ps[1], 'r')
ps = np.transpose(psd)
plt.plot(ps[0], ps[1], 'r')

plt.plot(close)
plt.show()










