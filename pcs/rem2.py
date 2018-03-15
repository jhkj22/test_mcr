def remove_2(ps):
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

while True:
    prev = len(ps)
    ps = remove_2(ps)
    ps = remove_n(ps)
    if len(ps) == prev:
        break
