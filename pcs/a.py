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
#close = close[70: 120]


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


ps_p = [[] for e in ps]
start = 0
if close[ps[start]] > close[ps[start + 1]]:
    start = 1
for i1 in range(start, len(ps) - 2, 2):
    mx = i1
    for i2 in range(i1 + 1, len(ps) - 1, 2):
        mx = max([mx, i2], key=lambda t: close[ps[t]])
        a = abs(close[ps[mx]] - close[ps[i1]]) / abs(close[ps[mx]] - close[ps[i2 + 1]])
        if a > 2.0:
            continue
        break
    ps_p[i1].append(mx)
if i1 <= len(ps) - 3:
    pass

start = len(ps) - 1
if close[ps[start]] > close[ps[start - 1]]:
    start = len(ps) - 2
for i1 in range(start, 1, -2):
    mx = i1
    for i2 in range(i1 - 1, 0, -2):
        mx = max([mx, i2], key=lambda t: close[ps[t]])
        a = abs(close[ps[mx]] - close[ps[i1]]) / abs(close[ps[mx]] - close[ps[i2 - 1]])
        if a > 2.0:
            continue
        break
    ps_p[i1].append(mx)
if i1 >= 3:
    pass

ps_ps = [[] for e in ps]
for i, o in enumerate(ps_p):
    if len(o) == 0:
        continue
    mi = min(o, key=lambda t: close[ps[t]])
    ps_ps[i].extend([abs(ps[mi] - ps[i]), close[ps[mi]] - close[ps[i]]])


ps_p = [[] for e in ps]
if close[ps[0]] < close[ps[1]]:
    start = 0
else:
    start = 1
for i1 in range(start, len(ps) - 2, 2):
    mx = close[ps[i1]]
    tmp_p = []
    for i2 in range(i1 + 1, len(ps) - 1, 2):
        mx = max([mx, close[ps[i2]]])
        a = abs(mx - close[ps[i1]]) / abs(mx - close[ps[i2 + 1]])
        if a < 0.5:
            break
        if a > 2.0:
            continue
        if len(tmp_p) == 0:
            tmp_p.append(i2 + 1)
            continue
        x = [ps[i1], ps[tmp_p[-1]], ps[i2 + 1]]
        y = close[x]
        a = (y[2] * (x[1] - x[0]) + y[0] * (x[2] - x[1])) / (x[2] - x[0])
        if a >= y[1]:
            continue
        a = (mx - y[1]) / (mx - a)
        if a < 0.85:
            tmp_p.append(i2 + 1)
    ps_p[i1].extend(tmp_p)


for i1, o1 in enumerate(ps_p):
    if len(o1) < 1:
        continue
    p1, s1 = ps[i1], ps_ps[i1][0]
    for i2 in range(len(o1) - 1, -1, -1):
        p2, s2 = ps[o1[i2]], ps_ps[o1[i2]][0]
        d = abs(p1 - p2)
        if d / s1 > 10 or d / s2 > 10:
            ps_p[i1].pop(i2)

"""
ps_p_av = [True if len(e) == 0 else False for e in ps_p]
ps_seq = []
def forward(l, i):
    if len(l) >= 2:
        a = abs(ps[l[-2]] - ps[l[-1]]) / abs(ps[l[-1]] - ps[i])
        if a <= 0.5 or a >= 2.0:
            if len(l) > 2:
                if len(ps_seq) == 0 or ps_seq[-1] != l:
                    ps_seq.append(l)
            return
        ps_p_av[i] = True
    l.append(i)
    if len(ps_p[i]) == 0:
        if len(l) > 2:
            ps_seq.append(l)
        return
    for o in ps_p[i]:
        forward(copy.deepcopy(l), o)
for i1 in range(len(ps_p)):
    if ps_p_av[i1]:
        continue
    forward([], i1)
"""

plt.plot(close)
"""
for i, o in enumerate(ps_ps):
    if len(o) < 1:
        continue
    if i == 0:
        continue
    if i == len(ps_ps) - 1:
        continue
    p, s = [ps[i], close[ps[i]]], o
    top = p[1] + s[1] / 2
    bot = p[1] - s[1] / 2
    left = p[0] - s[0] / 2
    right = p[0] + s[0] / 2
    p = [[left, right, right, left, left], [top, top, bot, bot, top]]
    plt.plot(p[0], p[1], color='orange')
"""

for i1, o in enumerate(ps_p):
    if len(o) < 1:
        continue
    for o in [[ps[i1], ps[i]] for i in o]:
        plt.plot(o, close[o], 'r')


plt.plot(ps, close[ps], 'ro')
for i, o in enumerate(ps):
    s = i % 10
    if s == 0:
        s = i
    plt.text(o, close[o], str(s))
plt.show()











