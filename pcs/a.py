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

start = getSize() - 10000 + 300
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

def ar_child(ps_p, i1):
    mx = i1
    tmp = []
    for i2 in range(i1 + 2, len(ps_p) - 1, 2):
        if ps_p[i2] < ps_p[i1]:
            break
        if ps_p[i2 + 1] > ps_p[mx]:
            mx = i2 + 1
            mn_i = i1 + 1 + np.argmin(ps_p[i1 + 1: mx])
            mn_v = ps_p[mn_i]
            mx_i = i1 + 1 + np.argmax(ps_p[i1 + 1: mn_i])
            mx_v = ps_p[mx_i]
            tmp.append([mx, (mx_v - mn_v) / (mx_v - ps[i1][1])])
    print(tmp)
    return tmp
def all_right(ps):
    ps_r = []
    ps_p = np.transpose(ps)[1]
    ps_pm = -ps_p
    if ps_p[0] < ps_p[1]:
        up = True
    else:
        up = False
    for i1 in range(len(ps) - 1):
        if up:
            ps_r.append(ar_child(ps_p, i1))
        else:
            ps_r.append(ar_child(ps_pm, i1))
        up = not up
    return ps_r


def remove_2(ps):
    while True:
        ps_r = []
        for i in range(len(ps) - 3):
            d = ps[i: i + 4, 1]
            d = np.abs(d[1:] - d[:-1])
            if d[1] * 2 > d[0] or d[1] * 2 > d[2]:
                continue
            ps_r.extend([i + 1, i + 2])
        if len(ps_r) == 0:
            break
        for i in reversed(ps_r):
            ps.pop(i)
    return ps
def remove_3(ps, close):
    for i in range(len(ps) - 4):
        d = ps[i: i + 5, ]
        d = np.abs(d[1:] - d[:-1])
        if d[0] < d[1] * 2 or d[3] < d[2] * 2:
            continue
        plt.plot(ps[i + 1: i + 4], close[ps[i + 1: i + 4]], 'ro')
    return ps

ps = remove_2(ps, close)
ps = remove_3(ps, close)

plt.plot(close)
for i, o in enumerate(ps):
    s = i % 10
    if s == 0:
        s = i
    plt.text(o[0], o[1], str(s))
ps = np.transpose(ps)
plt.plot(ps[0], ps[1], 'ro')
plt.show()










