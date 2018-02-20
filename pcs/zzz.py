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
        if a > 0.1:
            tmp_p.append(i2 + 1)
    ps_p[i1].extend(tmp_p)


ps_p = []
if close[ps[0]] < close[ps[1]]:
    start = 0
else:
    start = 1
for i1 in range(start, len(ps) - 2, 2):
    mx = close[ps[i1 + 1]]
    a = abs(mx - close[ps[i1]]) / abs(mx - close[ps[i1 + 2]])
    if a >= 0.5 and a <= 2.0:
        ps_p.append([ps[i1], ps[i1 + 2]])


ps_p_u = []
for i1 in range(len(ps_p) - 1):
    for i2 in range(i1 + 1, len(ps_p)):
        if ps_p[i1][1] < ps_p[i2][0]:
            break
        if ps_p[i1][1] == ps_p[i2][0]:
            ps_p_u.extend([i1, i2])
ps_p_u = list(set(ps_p_u))
print(ps_p_u)


sigma = 0.8
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


N = 1
ps = []
for n in range(len(close)):
    left = max([0, n - N])
    right = min([len(close), n + N + 1])
    if max(close[left: right]) == close[n]:
        ps.append([n, close[n]])
    elif min(close[left: right]) == close[n]:
        ps.append([n, close[n]])


class Block:
    def __init__(self):
        self.l = []
        self.type = 0
    def get_middle(self):
        mx = max(self.l, key=lambda t: t[1])[1]
        mn = min(self.l, key=lambda t: t[1])[1]
        return [(self.l[0][0] + self.l[-1][0]) / 2, (mx + mn) / 2]
    def draw(self):
        if len(self.l) == 0:
            return
        top = max(self.l, key=lambda t: t[1])[1]
        bot = min(self.l, key=lambda t: t[1])[1]
        left = self.l[0][0]
        right = self.l[-1][0]
        ps = [[left, right, right, left, left], [top, top, bot, bot, top]]
        plt.plot(ps[0], ps[1], color='orange')

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

#plt.subplot(121)
for f1 in range(1):
    [b.draw() for b in blocks]
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
plt.plot(ps[0], ps[1], 'r')
#plt.plot(close)
plt.show()



ps_n = []
for i in range(len(ps) - 3):
    n1 = abs(close[ps[i]] - close[ps[i + 1]])
    n2 = abs(close[ps[i + 1]] - close[ps[i + 2]])
    n3 = abs(close[ps[i + 2]] - close[ps[i + 3]])
    if n2 < n1 / 2 and n2 < n3 / 2:
        ps_n.extend([i + 1, i + 2])
for i in reversed(ps_n):
    ps.pop(i)

N = 7
M = 5
close = np.array(getClose(2 ** N))
plt.subplot(121)
plt.plot(close)
img = np.zeros((N * M, 2 ** N - 1))
#for itr in range(6):
#    close = [(close[n]+close[n+1])/2 for n in range(0, len(close), 2)]
for r in range(N):
    diff = close[1:] - close[:-1]
    for c in range(2 ** (N - r) - 1):
        img[r * M: (r + 1) * M, c * (2 ** r): (c + 1) * (2 ** r)] = diff[c]
    close = np.array([(close[n]+close[n+1])/2 for n in range(0, len(close), 2)])
plt.subplot(122)
plt.imshow(np.abs(img))
plt.show()

sigma = 3
filt_x = range(-2 * sigma, 2 * sigma + 1)
target_x = range(2 * sigma, len(close) - 2 * sigma)
X = np.array(filt_x)
Y = np.exp(-X ** 2 / 2 / sigma ** 2)

Y = Y / np.sum(Y)
g_filt = []
for n1 in target_x:
    s = 0
    for n2 in filt_x:
        s += close[n1 + n2] * Y[n2 + 2 * sigma]
    g_filt.append(s)
plt.plot(close)
plt.plot(target_x, g_filt)
plt.show()
