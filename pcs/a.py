import matplotlib.pyplot as plt
import numpy as np
import copy
from data import getSize, getClose
from extreme import get_ps, get_size, draw_ps
from smooth import gauss
import sys
sys.path.append('../')
from mathlib import exterior_angle

start = getSize() - 10000 + 1500
close = np.array(getClose(start, start + 2 ** 13))
"""
trades = []
have_pos = False
for i in range(2 ** 7, 2 ** 10):
    if not have_pos:
        if close[i] > close[i - 1]:
            trades.append([1, i])
            have_pos = True
    else:
        if abs(close[i] - close[trades[-1][1]]) > 0.0005:
            trades[-1].append(i)
            have_pos = False
if len(trades[-1]) == 2:
    trades.pop()

result_pos = []
result_cum = [0]

for t in trades:
    x = [t[1], t[2]]
    y = close[x]
    result_pos.append([x, y])
    result_cum.append(result_cum[-1] + y[1] - y[0])
    
display_pips = False
if display_pips:
    plt.plot(result_cum)
else:
    plt.plot(close[:2 ** 10])
    for x, y in result_pos:
        plt.plot(x, y, 'r')
plt.show()
"""
start = 00
close = close[start: start + 2 ** 8]

ps = get_ps(close)

def remove_2(ps, ps2):
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

def draw_ps(ps):
    for i, o in enumerate(ps):
        s = i % 10
        if s == 0:
            s = i
        plt.text(o[0], o[1], str(s))
    ps = np.transpose(ps)
    plt.plot(ps[0], ps[1], 'ro')
def get_2(ps):
    ps2 = []
    while True:
        prev = len(ps)
        ps = remove_2(ps, ps2)
        ps = remove_n(ps)
        if len(ps) == prev:
            break
    draw_ps(ps)
    return ps2
get_2(ps)

plt.plot(close)
plt.show()










