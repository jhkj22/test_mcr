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

start = getSize() - 10000 + 600
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

def drop(ps):
    if len(ps) > 3:
        return ps
    if len(ps) == 2:
        return []
    d = ps[:, 1]
    d = np.abs(d[1:] - d[:-1])
    a = min([d[0] / d[1], d[1] / d[0]])
    if a > 0.95:
        return [(ps[0] + ps[2]) / 2]
    if ps[0][1] > ps[1][1]:
        return [max(ps, key=lambda t: t[1])]
    else:
        return [min(ps, key=lambda t: t[1])]
def ar_child(ps_p, i1):
    mx = i1
    tmp = []
    for i2 in range(i1 + 1, len(ps_p)):
        if ps_p[i2] > ps_p[mx]:
            mx = i2
            tmp.append(i2)
        if i2 == len(ps_p) - 1:
            break
        if ps_p[i2 + 1] < ps_p[i1]:
            break
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
def tunnel(ps):
    print(all_right(ps))
def boxing(ps):
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
        a = drop(ps[ns])
        if len(a) > 1:
            tunnel(ps[[ns[0] - 1] + ns + [ns[-1] + 1]])
            continue
        for i in reversed(ns):
            ps = np.delete(ps, i, axis=0)
        if len(a) == 0:
            continue
        ps = np.concatenate((ps, a))
    tmp = []
    for o in ps:
        tmp.append([o[0], o[1]])
    ps = sorted(tmp, key=lambda t: t[0])
    return np.array(ps)

for i in range(1):
    ps = boxing(ps)



plt.plot(close)
ps = np.transpose(ps)
plt.plot(ps[0], ps[1], 'ro')
plt.show()










