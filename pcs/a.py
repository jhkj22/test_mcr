def getSize():
    f = open('../EURUSD.bin', 'rb')
    N = f.seek(0, 2) // 4
    f.close()
    return N
#f = open('C:/Users/hayato/AppData/Roaming/MetaQuotes/Terminal/50CA3DFB510CC5A8F28B48D1BF2A5702/history/FXDD-MT4 Demo Server/GBPJPY60.hst', 'rb')
#148, 60
def getClose(start, stop):
    f = open('../EURUSD.bin', 'rb')
    f.seek(start * 4, 0)
    l = f.read((stop - start) * 4)
    f.close()
    import struct
    l = struct.unpack_from(">"+"f"*(stop-start), l)
    return l

import matplotlib.pyplot as plt
import numpy as np

start = getSize() - 10000# - 500
close = np.array(getClose(start, start + 2 ** 8))
#close = close[:50]

class Block:
    def __init__(self):
        self.l = []
        self.type = 0
    def get_middle(self):
        mx = max(self.l, key=lambda t: t[1])[1]
        mn = min(self.l, key=lambda t: t[1])[1]
        return [(self.l[0][0] + self.l[-1][0]) / 2, (mx + mn) / 2]

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
def  get_extreme(blocks):
    ps = [blocks[0].get_middle()]
    for i in range(1, len(blocks) - 1):
        m = blocks[i - 1].get_middle()
        c = blocks[i].get_middle()
        p = blocks[i + 1].get_middle()
        if c[1] < p[1] and c[1] < m[1]:
            ps.append(c)
        elif c[1] > p[1] and c[1] > m[1]:
            ps.append(c)
    ps.append(blocks[-1].get_middle())
    return ps
blocks = []
for n in first(close):
    b = Block()
    b.l.append([n, close[n]])
    blocks.append(b)

for f1 in range(4):
    ps = get_extreme(blocks)
    ps = np.array(ps)
    
    ps_n = []
    tmp = []
    for i in range(len(ps) - 2):
        n1 = abs(ps[i][1] - ps[i + 1][1])
        n2 = abs(ps[i + 1][1] - ps[i + 2][1])
        if n2 < n1 / 2:
            tmp = [i + 1]
        elif n1 < n2 / 2:
            if len(tmp) != 0:
                tmp.append(i + 1)
                ps_n.append(tmp)
                tmp = []
        else:
            if len(tmp) != 0:
                tmp.append(i + 1)
    blocks = []
    i = 0
    while i < len(ps):
        if len(ps_n) > 0:
            if ps_n[0][0] == i:
                b = Block()
                b.l = ps[ps_n[0]]
                blocks.append(b)
                i = ps_n[0][-1] + 1
                ps_n.pop(0)
                continue
        b = Block()
        b.l.append(list(ps[i]))
        blocks.append(b)
        i += 1


ps = np.transpose(ps)
plt.plot(ps[0], ps[1], 'ro')
plt.plot(close)
plt.show()














