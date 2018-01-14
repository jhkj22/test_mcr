def edge(line):
    l = [[0, -1]]
    for i in range(1, len(line)):
        if line[i] >= line[i - 1]:
            l[-1][1] = i
        else:
            if l[-1][1] < 0:
                l[-1][0] = i
            else:
                l.append([i, -1])
    if l[-1][1] < 0:
        l.pop(-1)
    l = [o for o in l if abs(line[o[0]] - line[o[1]]) > 30]
    l = [o[0] for o in l if o[1] - o[0] > 30]
    return l





