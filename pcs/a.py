def getSize():
    f = open('./EURUSD.bin', 'rb')
    N = f.seek(0, 2) // 4
    f.close()
    return N
def getClose(start, stop):
    f = open('./EURUSD.bin', 'rb')
    f.seek(start * 4, 0)
    l = f.read((stop - start) * 4)
    f.close()
    import struct
    l = struct.unpack_from(">"+"f"*(stop-start), l)
    return l

import matplotlib.pyplot as plt
import numpy as np
import copy

start = getSize() - 10000 + 500
close = np.array(getClose(start, start + 2 ** 7))

def first(close):
    tmp = []
    ps = []
    for i in range(len(close) - 1):
        if close[i] == close[i + 1]:
            if len(tmp) == 0:
                tmp = [i, i + 1]
            else:
                tmp.append(i + 1)
        else:
            if len(tmp) != 0:
                ps.append(tmp[len(tmp) // 2])
                tmp = []
            else:
                ps.append(i)
    if len(tmp) != 0:
        ps.append(tmp[len(tmp) // 2])
        tmp = []
    else:
        ps.append(i + 1)
    return ps
def  get_extreme(ps, close):
    l = [ps[0]]
    for i in range(1, len(ps) - 1):
        m = close[ps[i - 1]]
        c = close[ps[i]]
        p = close[ps[i + 1]]
        if c < p and c < m:
            l.append(ps[i])
        elif c > p and c > m:
            l.append(ps[i])
    l.append(ps[-1])
    return l

ps = first(close)
ps = get_extreme(ps, close)
ps = np.transpose([ps, close[ps]])

def remove_2(ps):
    while True:
        rm = []
        for i in range(len(ps) - 3):
            d = ps[i: i + 4, 1]
            d = np.abs(d[1:] - d[:-1])
            a = [d[0] / d[1], d[2] / d[1]]
            if a[0] > 1.05 and a[1] > 1.05:
                rm.extend([i + 1, i + 2])
        if len(rm) == 0:
            break
        for i in rm[::2]:
            plt.plot([ps[i][0], ps[i + 1][0]], [ps[i][1], ps[i + 1][1]], 'r', linewidth=5)
        ps = np.delete(ps, rm, axis=0)
    return ps

def drop(ps):
    if len(ps) % 2 == 0:
        return []
    x = (ps[1][0] + ps[-2][0]) // 2
    if ps[0][1] > ps[1][1]:
        y = min(ps, key=lambda t: t[1])[1]
    else:
        y = max(ps, key=lambda t: t[1])[1]
    return [[x, y]]

def remove_n(ps):
    ps_n = []
    tmp = []
    for i in range(len(ps) - 2):
        d = ps[i: i + 3, 1]
        d = np.abs(d[1:] - d[:-1])
        if d[1] < d[0] / 1.8:
            tmp = [i + 1]
        elif d[0] < d[1] / 1.8:
            if len(tmp) != 0:
                tmp.append(i + 1)
                ps_n.append(tmp)
                tmp = []
        else:
            if len(tmp) != 0:
                tmp.append(i + 1)
    for ns in reversed(ps_n):
        a = drop(ps[[ns[0] - 1] + ns + [ns[-1]]])
        for i in reversed(ns):
            ps = np.delete(ps, i, axis=0)
        ps = np.concatenate((ps, a))
    tmp = []
    for o in ps:
        tmp.append([o[0], o[1]])
    ps = np.array(sorted(tmp, key=lambda t: t[0]))
    return ps

while True:
    prev = len(ps)
    ps = remove_2(ps)
    ps = remove_n(ps)
    if len(ps) == prev:
        break


plt.plot(close)
ps = np.transpose(ps)
plt.plot(ps[0], ps[1], 'ro')
plt.show()










