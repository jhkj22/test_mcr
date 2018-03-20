import matplotlib.pyplot as plt
import numpy as np
import copy
from data import getSize, getClose
from extreme import get_ps

start = getSize() - 10000 + 200
close = np.array(getClose(start, start + 2 ** 7))
close = close[30: 50]

ps = get_ps(close)

def get_size():
    ps_size = [[ps[1][0] - ps[0][0], abs(ps[1][1] - ps[0][1])]]
    for i in range(1, len(ps) - 1):
        l = ps[i - 1][1]
        c = ps[i][1]
        r = ps[i + 1][1]
        ld, rd = abs(l - c), abs(r - c)
        if ld < rd:
            w = ps[i][0] - ps[i - 1][0]
            h = ld
        else:
            w = ps[i + 1][0] - ps[i][0]
            h = rd
        ps_size.append([w, h])
    ps_size.append([ps[i + 1][0] - ps[i][0], abs(ps[i + 1][1] - ps[i][1])])
    return ps_size

ps_size = get_size()

def get_up(ps, i1):
    pp = []
    mx = ps[i1][1]
    tmp = []
    for i2 in range(i1 + 1, len(ps) - 1, 2):
        #if ps[i2 + 1][0] - ps[i1][0] > ps_size[i1][0] * 5 or \
        #   ps[i2 + 1][0] - ps[i1][0] > ps_size[i2 + 1][0] * 5:
        #    break
        mx = max([mx, ps[i2][1]])
        if len(tmp) > 0:
            aa = []
            for o3 in tmp:
                x = [ps[i1][0], ps[o3][0], ps[i2 + 1][0]]
                y = [ps[i1][1], ps[o3][1], ps[i2 + 1][1]]
                aa.append((y[2] * (x[1] - x[0]) + y[0] * (x[2] - x[1])) / (x[2] - x[0]))
            if aa[-1] >= y[1]:
                continue
            f = False
            for n3 in range(len(tmp)):
                a = (mx - ps[tmp[n3]][1]) / (mx - aa[n3])
                if a > 0.85:
                    f = True
                    break
            if f:
                tmp.append(i2 + 1)
                continue
        tmp.append(i2 + 1)
        a = abs(mx - ps[i1][1]) / abs(mx - ps[i2 + 1][1])
        if a < 0.5:
            break
        if a > 2.0:
            continue
        pp.append(i2 + 1)
    return pp

def get_node():
    ps_p = [[] for e in ps]
    ps_m = np.array([[o[0], -o[1]] for o in ps])
    flip = True
    if ps[0][1] < ps[1][1]:
        flip = False
    for i1 in range(len(ps) - 2):
        if flip:
            ps_p[i1].extend(get_up(ps_m, i1))
        else:
            ps_p[i1].extend(get_up(ps, i1))
        flip = not flip
    return ps_p

ps_p = get_node()
print(ps_p)

def draw_ps(ps):
    for i, o in enumerate(ps):
        s = i % 10
        if s == 0:
            s = i
        plt.text(o[0], o[1], str(s))
    ps = np.transpose(ps)
    plt.plot(ps[0], ps[1], 'ro')
plt.plot(close)
draw_ps(ps)

plt.show()










