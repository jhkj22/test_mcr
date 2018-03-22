import numpy as np


def gauss(close, sigma):
    sigma_n = int(np.ceil(2 * sigma))
    filt_x = range(-sigma_n, sigma_n + 1)
    X = np.array(filt_x)
    Y = np.exp(-X ** 2 / 2 / sigma ** 2)
    Y = Y / np.sum(Y)
    g_filt = []
    for n1 in range(len(close)):
        s = 0
        for n2 in filt_x:
            if n1 + n2 < 0:
                cl = close[0]
            elif n1 + n2 >= len(close):
                cl = close[-1]
            else:
                cl = close[n1 + n2]
            s += cl * Y[n2 + sigma_n]
        g_filt.append(s)
    return np.array(g_filt)





