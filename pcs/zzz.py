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
class Tunnel:
    def __init__(self, ps):
        self.path = all_right(ps)
        self.ps = ps
        self.debug = False
        if ps[0][0] == 3:
            self.debug = False
    def all_path(self):
        self.whole = []
        self.__rec(len(self.path), [0])
        return self.whole
    def is_dangling(self):
        t = 0
        self.whole = []
        self.__rec(len(self.path) - 1, [0])
        self.whole = [o for o in self.whole if o[-1] == len(self.path) - 1]
        if len(self.whole) > 0:
            return False
        self.whole = []
        self.__rec(len(self.path), [1])
        self.whole = [o for o in self.whole if o[-1] == len(self.path)]
        if len(self.whole) > 0:
            return False
        return True
    def __rec(self, stop, top):
        if self.debug:
            print(top)
        if len(top) >= 3:
            pp = [self.ps[top[n]][1] for n in range(-3, 0)]
            d = [abs(pp[0] - pp[1]), abs(pp[1] - pp[2])]
            a = max([d[0] / d[1], d[1] / d[0]])
            if a > 1.8:
                return
        if top[-1] >= stop:
            top = [o for o in top if o <= stop]
            if len(top) >= 3:
                self.whole.append(copy.deepcopy(top))
            return
        for p in self.path[top[-1]]:
            self.__rec(stop, top + [p])
def drop(ps):
    if len(ps) > 3:
        return ps
    """
        tn = Tunnel(ps)
        d = tn.is_dangling()
        if d:
            if len(ps) % 2 == 0:
                return []
            else:
                mx = max(ps, key=lambda t: t[1])
                mn = min(ps, key=lambda t: t[1])
                if mx[0] == ps[0][0] or mx[0] == ps[len(ps) - 1][0]:
                    return [mn]
                else:
                    return [mx]
        else:
            return ps[1:-1]"""
    #ps = ps[1:-1]
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
def boxing(ps):
    for outer_e in range(10):
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
        fin = True
        for ns in reversed(ps_n):
            #a = drop(ps[[ns[0] - 1] + ns + [ns[-1] + 1]])
            a = drop(ps[ns])
            if len(a) < len(ns):
                fin = False
            for i in reversed(ns):
                ps = np.delete(ps, i, axis=0)
            if len(a) == 0:
                continue
            ps = np.concatenate((ps, a))
        tmp = []
        for o in ps:
            tmp.append([o[0], o[1]])
        ps = np.array(sorted(tmp, key=lambda t: t[0]))
        if fin:
            break
    return ps






def remove_2(ps, close):
    while True:
        ps_r = []
        for i in range(len(ps) - 3):
            d = close[ps[i: i + 4]]
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
        d = close[ps[i: i + 5]]
        d = np.abs(d[1:] - d[:-1])
        if d[0] < d[1] * 2 or d[3] < d[2] * 2:
            continue
        plt.plot(ps[i + 1: i + 4], close[ps[i + 1: i + 4]], 'ro')
    return ps

ps = remove_2(ps, close)
ps = remove_3(ps, close)


plt.plot(ps, close[ps], 'ro')
for i, o in enumerate(ps):
    s = i % 10
    if s == 0:
        s = i
    plt.text(o, close[o], str(s))

for i1, o in enumerate(ps_p):
    if len(o) < 1:
        continue
    for o in [[ps[i1], ps[i]] for i in o]:
        plt.plot(o, close[o], 'r')

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


def blocking():
    N = 2
    ps_rem = []
    for i in range(1, len(ps) - N):
        mx = max(close[ps[i: i + N]])
        mn = min(close[ps[i: i + N]])
        height = mx - mn
        left_i = i - 1
        right_i = i + N
        left = close[ps[left_i]]
        right = close[ps[right_i]]
        left_node = max([abs(mx - left), abs(mn - left)])
        right_node = max([abs(mx - right), abs(mn - right)])
        if left_node > height * 2 and right_node > height * 2:
            ps_rem.extend(range(i, i + N))
    for i in reversed(ps_rem):
        ps.pop(i)

blocking()

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

def get_up(close, i1):
    pp = []
    mx = close[ps[i1]]
    tmp = []
    for i2 in range(i1 + 1, len(ps) - 1, 2):
        if ps[i2 + 1] - ps[i1] > ps_size[i1][0] * 5 or \
           ps[i2 + 1] - ps[i1] > ps_size[i2 + 1][0] * 5:
            break
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
        pp.append(i2 + 1)
    return pp

def get_node():
    ps_p = [[] for e in ps]
    flip = True
    if close[ps[0]] < close[ps[1]]:
        flip = False
    for i1 in range(len(ps) - 2):
        if flip:
            ps_p[i1].extend(get_up(-close, i1))
        else:
            ps_p[i1].extend(get_up(close, i1))
        flip = not flip
    return ps_p

ps_p = get_node()

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



import networkx

vector = {}
for i, ps in enumerate(ps_p):
    if len(ps) == 0:
        continue
    vector[i] = ps

graph = networkx.Graph(vector)
pos = networkx.spring_layout(graph)

networkx.draw_networkx_nodes(graph, pos)
networkx.draw_networkx_edges(graph, pos)
networkx.draw_networkx_labels(graph, pos)


#--------size------------
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

for i1, o1 in enumerate(ps_p):
    if len(o1) < 1:
        continue
    p1, s1 = ps[i1], ps_ps[i1][0]
    for i2 in range(len(o1) - 1, -1, -1):
        p2, s2 = ps[o1[i2]], ps_ps[o1[i2]][0]
        d = abs(p1 - p2)
        if d / s1 > 10 or d / s2 > 10:
            ps_p[i1].pop(i2)


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
#------------------------




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
