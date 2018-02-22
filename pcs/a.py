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

start = getSize() - 10000 + 400
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

def get_size():
    ps_size = [[ps[1] - ps[0], close[ps[1]] - close[ps[0]]]]
    for i in range(1, len(ps) - 1):
        l = close[ps[i - 1]]
        c = close[ps[i]]
        r = close[ps[i + 1]]
        ld, rd = abs(l - c), abs(r - c)
        if ld < rd:
            w = ps[i] - ps[i - 1]
            h = ld
        else:
            w = ps[i + 1] - ps[i]
            h = rd
        ps_size.append([w, h])
    ps_size.append([ps[i + 1] - ps[i], close[ps[i + 1]] - close[ps[i]]])
    return ps_size

ps_size = get_size()

def get_up():
    mx = close[ps[i1]]
    tmp = []
    for i2 in range(i1 + 1, len(ps) - 1, 2):
        mx = max([mx, close[ps[i2]]])
        if len(tmp) > 0:
            aa = []
            for o3 in tmp:
                x = [ps[i1], ps[o3], ps[i2 + 1]]
                y = close[x]
                aa.append((y[2] * (x[1] - x[0]) + y[0] * (x[2] - x[1])) / (x[2] - x[0]))
            if aa[-1] >= y[1]:
                continue
            f = False
            for n3 in range(len(tmp)):
                a = (mx - close[ps[tmp[n3]]]) / (mx - aa[n3])
                if a > 0.85:
                    f = True
                    break
            if f:
                tmp.append(i2 + 1)
                continue
        tmp.append(i2 + 1)
        a = abs(mx - close[ps[i1]]) / abs(mx - close[ps[i2 + 1]])
        if a < 0.5:
            break
        if a > 2.0:
            continue
        ps_p[i1].append(i2 + 1)
def get_down():
    pass

def get_node():
    ps_p = [[] for e in ps]
    for i1 in range(len(ps) - 2):
        get_up()
        get_down()



plt.plot(close)

"""
for i1, o in enumerate(ps_p):
    if len(o) < 1:
        continue
    for o in [[ps[i1], ps[i]] for i in o]:
        plt.plot(o, close[o], 'r')
"""
for i, o in enumerate(ps_size):
    #if i == 0 or i == len(ps_size) - 1:
    #    continue
    p, s = [ps[i], close[ps[i]]], o
    top = p[1] + s[1] / 2
    bot = p[1] - s[1] / 2
    left = p[0] - s[0] / 2
    right = p[0] + s[0] / 2
    p = [[left, right, right, left, left], [top, top, bot, bot, top]]
    plt.plot(p[0], p[1], color='orange')

plt.plot(ps, close[ps], 'ro')
for i, o in enumerate(ps):
    s = i % 10
    if s == 0:
        s = i
    plt.text(o, close[o], str(s))

plt.show()










