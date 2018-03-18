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
    def all_path(self):
        self.whole = []
        self.visited = [False for e in self.ps]
        for n in range(len(ps) - 2):
            if self.visited[n]:
                continue
            self.part = []
            self.__rec2([n])
            self.whole.append(self.part)
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
    def __rec2(self, top):
        if len(top) >= 3:
            pp = [self.ps[top[n]][1] for n in range(-3, 0)]
            d = [abs(pp[0] - pp[1]), abs(pp[1] - pp[2])]
            a = max([d[0] / d[1], d[1] / d[0]])
            if a > 1.8:
                if len(self.part) == 0 or self.part[-1] < top[-1]:
                    self.part = top
                return
            self.visited[top[-2]] = True
        if top[-1] >= len(self.path) - 1:
            if len(self.part) == 0 or self.part[-1] < top[-1]:
                self.part = top
            return
        for p in self.path[top[-1]]:
            self.__rec2(top + [p])
def drop(ps):
    if len(ps) > 5:
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
            return ps[1:-1]
    ps = ps[1:-1]
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
            a = drop(ps[[ns[0] - 1] + ns + [ns[-1] + 1]])
            #a = drop(ps[ns])
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

ps = boxing(ps)
tn = Tunnel(ps)
a = tn.all_path()
