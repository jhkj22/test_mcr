store = []
tmp = []
points = []

for r in range(img.shape[0]):
    line = img[r].astype(np.int)
    diff = np.abs(line[:-1] - line[1:])
    l = []
    for c in range(1, len(diff) - 1):
        if diff[c] < 20:
            continue
        if diff[c] > diff[c - 1] and diff[c] > diff[c + 1]:
            l.append(c)
            points.append([r, c])
    i, j = 0, 0
    while i < len(tmp) and j < len(l):
        if np.abs(tmp[i][-1][1] - l[j]) <= 1:
            tmp[i].append([r, l[j]])
            l.pop(j)
            i += 1
        elif tmp[i][-1][1] < l[j]:
            i += 1
        else:
            j += 1
    i = len(tmp) - 1
    while i >= 0:
        if tmp[i][-1][0] != r:
            store.append(tmp[i])
            tmp.pop(i)
        i -= 1
    for o in l:
        tmp.append([[r, o]])
    tmp = sorted(tmp, key=lambda t: t[-1][1])
for o in tmp:
    store.append(o)

def tangent_p(head, tail):
    index = []
    for i1 in range(3):
        for i2 in range(3):
            index.append([i1, i2])
    index = sorted(index, key=lambda t: t[0] + t[1])
    for (i1, i2) in index:
        i2 = -i2 - 1
        if tail[i2][0] >= head[i1][0]:
            continue
        if abs(head[i1][1] - tail[i2][1]) > abs(head[i1][0] - tail[i2][0]):
            continue
        ang_h = exterior_angle([head[i1 + 1], head[i1], tail[i2]])
        ang_t = exterior_angle([tail[i2 - 1], tail[i2], head[i1]])
        if ang_h < 90 and ang_t < 90:
            return [i1, i2]
    return []


i1 = 0
while i1 < len(store):
    o1 = store[i1]
    for i2 in range(len(store) - 1, -1, -1):
        o2 = store[i2]
        if i1 == i2:
            continue
        if len(o1) < 4 or len(o2) < 4:
            continue
        if abs(o1[0][0] - o2[-1][0]) > 3:
            continue
        if abs(o1[0][1] - o2[-1][1]) > 3:
            continue
        joint_i = tangent_p(o1, o2)
        if joint_i == []:
            continue
        if joint_i[1] < -1:
            store[i2] = store[i2][:joint_i[1] + 1]
        store[i2].extend(store[i1][joint_i[0]:])
        store[i1] = store[i2]
        store.pop(i2)
        i1 -= 1
        break
    i1 += 1

for ps in store:
    if len(ps) < 5:
        continue
    ps = np.transpose(ps)
    plt.plot(ps[0], ps[1], 'r')

points = np.transpose(points)
plt.plot(points[0], points[1], 'bo', markersize=1)
plt.imshow(img.T, cmap='gray', vmin=0, vmax=256)
