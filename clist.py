def get(l, *acc):
    if len(acc) == 1:
        return l[acc[0] % len(l)]
    elif len(acc) == 2:
        tl = []
        n = acc[0]
        while True:
            if n == acc[1]:
                break
            tl.append(l[n % len(l)])
            n += 1
        return tl
    elif len(acc) == 3:
        tl = []
        for n in range(acc[0], acc[1], acc[2]):
            tl.append(l[n % len(l)])
        return tl
    else:
        raise Exception

def range(l, *acc):
    def abc(n):
        n %= len(l)
        if n < 0:
            n += len(l)
        return n
    if len(acc) == 1:
        return [abc(acc[0])]
    elif len(acc) == 2:
        tl = []
        n = acc[0]
        e = acc[1]
        while True:
            if n == e:
                break
            tl.append(abc(n))
            n += 1
        return tl
    elif len(acc) == 3:
        tl = []
        for n in range(acc[0], acc[1], acc[2]):
            tl.append(l[n % len(l)])
        return tl
    else:
        raise Exception










