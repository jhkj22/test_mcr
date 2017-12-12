dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
dy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
pw = np.sqrt(dx ** 2 + dy ** 2).astype(np.uint)
def quant_th(tht):
    thq = np.copy(tht)
    for i in range(-3, 4):
        thq[(np.pi*(2*i-1)/8 <= tht) & (tht < np.pi*(2*i+1)/8)] = i
    thq = np.abs(thq)
    thq[(tht >= np.pi*7/8) | (tht < -np.pi*7/8)] = 0
    return thq

tht = np.arctan2(dx, -dy)
thq = quant_th(tht)
plt.imshow(thq, cmap='gray')
plt.show()
ps_img = np.zeros((70, 70))
ps = [[60, 30]]; ps_img[60, 30] = 1
for n1 in range(6):
    r, c = ps[-1]
    I = pw[r - 1: r + 2, c - 1: c + 2]
    l = np.argsort(I.flatten())[::-1]
    for o2 in l:
        rr, cc = r + o2 // 3 - 1, c + o2 % 3 - 1
        if ps_img[rr, cc] == 1:
            continue
        ps_img[rr, cc] = 1
        ps.append([rr, cc])
        break
ps = np.transpose(np.array(ps) - np.array([40, 20]))
plt.plot(ps[1], ps[0], 'ro')
plt.imshow(pw[40:, 20:50], cmap='gray')
plt.show()
ld = []
for r in range(1, 69):
    for c in range(1, 69):
        I = pw[r - 1: r + 2, c - 1: c + 2]
        l = np.argsort(I.flatten())[::-1]
        q = list(l[:3])
        if not 4 in q:
            continue
        q.remove(4)
        ae = lambda t: [t // 3, t % 3]
        v1 = np.array(ae(q[0]))
        v2 = np.array(ae(q[1]))
        nv = np.linalg.norm(v1 - v2)
        if nv >= 2:
            #line_dot[r, c] = 1
            ld.append([r, c])
ld = np.transpose(ld)
#plt.plot(ld[1], ld[0], 'ro')
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(pw, cmap='gray')
plt.show()
l = []
img_l = np.zeros((69, 69))
for r in range(69):
    for c in range(69):
        I = img[r: r + 2, c: c + 2]
        mm = np.max(I) - np.min(I)
        if mm < 20:
            img_l[r, c] = 1
            l.append([r, c])
l = np.transpose(l)


img = cv2.pyrMeanShiftFiltering(img, 32, 32)

out = np.zeros((69, 69))
l = []
for r in range(69):
    for c in range(69):
        I = img[r: r + 2, c: c + 2]
        mx, mn = np.max(I), np.min(I)
        out[r, c] = mx - mn
        if mx - mn > 50:
            l.append([r, c])
#l = np.transpose(l, (1, 0))
plt.imshow(out, cmap='gray')
#plt.plot(l[1], l[0], 'ro')
plt.show()



img = img.astype(np.int)
l = []

for r in range(70):
    for c in range(1, 70):
        if abs(line[r, c - 1] - line[r, c]) > 40:
            l.append([r, c])
for c in range(11, 12):
    for r in range(1, 70):
        if abs(img[r - 1, c] - img[r, c]) > 40:
            l.append([r, c])
l = np.transpose(l, (1, 0))
plt.subplot(131)
plt.plot(l[1], l[0], 'ro')
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.imshow(img[:,11:12], cmap='gray')
plt.axis('off')
plt.subplot(133)
plt.plot(range(70), img[:,11].flatten())

ps_u = []
ps_l = []
for r in range(1, 69):
    for c in range(1, 69):
        I = img[r - 1: r + 2, c - 1: c + 2]
        mx, mn = np.max(I), np.min(I)
        if mx - mn > 241:
            ps_u.append([r, c])
ps = np.transpose(ps_u, (1, 0))
plt.plot(ps[1], ps[0], 'ro')



p = np.ones((100, 50))
plt.imshow(np.concatenate((p * 129, p * 130), axis=1), cmap='gray', vmin=0, vmax=256)


x, y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
flat = np.ones((100, 50)) * 10
plt.imshow(np.concatenate((x, flat, np.flip(x, axis=1)), axis=1), cmap='gray', vmin=-10, vmax=20)


l = []
    for d in diff:
        if d > 10:
            l.append(1)
        elif d < -10:
            l.append(-1)
        else:
            l.append(0)
    l2 = [[l[0], 0, 0]]
    for n in range(1, len(l)):
        o = l[n]
        if l2[-1][0] == o:
            l2[-1][2] = n
        else:
            l2.append([o, n, n])
    l2 = [(o[1] + o[2]) // 2 for o in l2 if o[0] != 0]



def is_flat(r, c):
    block = img[r: r + 2, c: c + 2]
    return (np.max(block) - np.min(block)) < 50
ps = []
for r in range(0, 50, 2):
    for c in range(0, 50, 2):
        if not is_flat(r, c):
            ps.append([r, c])
ys, xs = np.transpose(ps, (1, 0)) + 0.5
plt.plot(xs, ys, 'ro')



n = 20
a = np.array([100 for e in range(n)])
shade = np.arange(0, 100, 50 / n)
b = np.array([100 for e in range(n)]) - shade[:n]
c = np.array([200 for e in range(n)]) - shade[n:]
d = np.array([100 for e in range(n)])
e = np.concatenate((a, b, c, d))
plt.imshow([e], cmap='gray', vmin=0, vmax=256)
plt.axis('off')
plt.show()


out = np.zeros(img.shape)
for n in range(0, 256, 8):
    out[(n <= img) & (img < n + 8)] = n

from urllib.request import urlopen

with open('../fall11_urls.txt', 'r') as f:
    for n in range(14):
        s = f.readline()
url = s.split('\n')[0].split('\t')[1]

from PIL import Image
import requests
from io import BytesIO
response = requests.get(url)
img = Image.open(BytesIO(response.content))


directions[(directions.index([sa[0], sa[1]]) + 1) % len(directions)]

def scan(ps):
    center = []
    count = 0
    for i, p in enumerate(ps):
        if img[p[0], p[1]] == 1:
            count += 1
        else:
            if count > 0:
                center.append([i - (count // 2 + 1), count])
                count = 0
    return center

def thickness_1d(edge, d):
    comp = []; uncomp = []
    for s in edge:
        line = []
        while np.all((0 <= s) & (s < img.shape[1])):
            line.append(np.copy(s))
            s += d
        head = [[line[i], s] for i, s in scan(line)]
        j = len(uncomp) - 1
        while j >= 0:
            i = len(head) - 1
            cont = False
            while i >= 0:
                if np.max(np.abs(uncomp[j][-1][0] - head[i][0])) < 3 and abs(uncomp[j][-1][1] - head[i][1]) < 4:
                    uncomp[j].append(head[i])
                    head.pop(i)
                    cont = True
                i -= 1
            if not cont:
                comp.append(uncomp[j])
                uncomp.pop(j)
            j -= 1
        for h in head:
            uncomp.append([h])
    comp = [c for c in comp if len(c) >= min(c, key=lambda t: t[1])[1]]
    return comp
def thickness():
    start = []
    for r in range(img.shape[0]):
        start.append([r, 0])
    return thickness_1d(np.array(start), np.array([0, 1]))

def plot_comp(comp, cl):
    for c in comp:
        x, y = [], []
        for p in c:
            x.append(p[0][1]); y.append(p[0][0])
        plt.plot(x, y, cl)
img_o = np.copy(img)
for i, theta in enumerate([0, 45, 90, 135]):
    img = ndimage.rotate(img_o, theta)
    comp = thickness()
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plot_comp(comp, 'r')

def perimeter():
    lt = []
    for r in range(1, 5):
        rd0 = circle(0, 0, r)
        rd1 = circle(0, 0, r + 1)
        rd0 = np.transpose(rd0, (1, 0))
        rd1 = np.transpose(rd1, (1, 0))
        rd0 = [tuple(o) for o in rd0]
        rd1 = [tuple(o) for o in rd1]
        rd = set(rd1) - set(rd0)
        rd = list(rd)
        norm = lambda t: t[0] ** 2 + t[1] ** 2
        rd = sorted(rd, key=norm)
        tmp = []
        start = 0
        for i in range(1, len(rd)):
            if norm(rd[i]) != norm(rd[i - 1]):
                tmp.append(rd[start: i])
                start = i
        tmp.append(rd[start: len(rd)])
        for o in tmp:
            lt.append(o)
    lts = []
    for o in lt:
        lts.append(sorted(o, key=lambda t: np.arctan2(t[1], t[0])))
    return lts

rds = perimeter()
img_s = np.zeros((28, 28))

for r in range(28):
    for c in range(28):
        if img[r, c] == 0:
            continue
        flag = False
        for s, rd in enumerate(rds):
            half = len(rd) // 2
            for i in range(half):
                a1, a2 = rd[i], rd[(i + half) % len(rd)]
                if img[r + a1[0], c + a1[1]] == 0 and img[r + a2[0], c + a2[1]] == 0:
                    img_s[r, c] = s + 1
                    plt.plot([c + a1[1], c + a2[1]], [r + a1[0], r + a2[0]], 'r')
                    flag = True
            if flag:
                break
per = [[1, 0], [0, 1], [-1, 0], [-1, 0], [1, -1], [-1, 1], [-1, -1], [1, 1]]
img_t = np.zeros((28, 28))
for r in range(28):
    for c in range(28):
        if img_s[r, c] == 0:
            continue
        l = []
        for p in per:
            v = img_s[r + p[0], c + p[1]]
            if v > 0:
                l.append(v)
        if img_s[r, c] <= min(l):
            img_t[r, c] = img_s[r, c]
plt.subplot(121)
plt.imshow(img_s)
plt.subplot(122)

def rect_contour():
    rc = []
    for r in range(28):
        rc.append([r, 0])
    for c in range(1, 28):
        rc.append([27, c])
    for r in range(26, -1, -1):
        rc.append([r, 27])
    for c in range(26, 0, -1):
        rc.append([0, c])
    return np.array(rc)
rect_c = rect_contour()
seek_info = {'v': {'start': 81, 'stop': 112, 'dir': np.array([1, 0])},
             'h': {'start': 0, 'stop': 28, 'dir': np.array([0, 1])},
             'ru': {'start': 0, 'stop': 56, 'dir': np.array([-1, 1])},
             'lu': {'start': 28, 'stop': 84, 'dir': np.array([-1, -1])}}
def thickness(key):
    t = seek_info[key]
    return thickness_1d(rect_c[t['start']: t['stop']], t['dir'])



def perimeter(r):
    rd0 = circle(0, 0, r)
    rd1 = circle(0, 0, r + 1)
    rd0 = np.transpose(rd0, (1, 0))
    rd1 = np.transpose(rd1, (1, 0))
    rd0 = [tuple(o) for o in rd0]
    rd1 = [tuple(o) for o in rd1]
    rd = set(rd1) - set(rd0)
    rd = list(rd)
    rd = sorted(rd, key=lambda t: np.arctan2(t[1], t[0]))
    return np.array(rd)
rds = [perimeter(r) for r in range(1, 5)]
img_s = np.zeros((28, 28))
for r in range(28):
    for c in range(28):
        if img[r, c] == 0:
            continue
        flag = False
        for s, rd in enumerate(rds):
            half = len(rd) // 2
            for i in range(half):
                a1, a2 = rd[i], rd[(i + half) % len(rd)]
                if img[r + a1[0], c + a1[1]] == 0 and img[r + a2[0], c + a2[1]] == 0:
                    img_s[r, c] = s + 1
                    flag = True
                    break
            if flag:
                break



def centre(p):
    double_k = lambda r: r[0] ** 2 + r[1] ** 2
    def is_opposite(p1, p2):
        a = np.arctan2(p1[1], p1[0])
        b = np.arctan2(p2[1], p2[0])
        d = np.abs(a - b)
        if 2 * np.pi / 3 < d and d < 4 * np.pi / 3:
            return True
        else:
            return False
    nearests = []
    seconds = []
    tmp = nearests
    break_next = False
    for rd in rds:
        if nearests != []:
            break_next = True
        k = np.inf
        for r in rd:
            if double_k(r) > k:
                tmp = seconds
            if sur[p[0] + r[0], p[1] + r[1]] == 1:
                k = double_k(r)
                tmp.append(r)
        if break_next:
            break
    for n1 in range(len(nearests)):
        for n2 in range(n1 + 1, len(nearests)):
            if is_opposite(nearests[n1], nearests[n2]):
                plt.plot(p[1], p[0], 'ro')
                return
    for o1 in nearests:
        for o2 in seconds:
            if is_opposite(o1, o2):
                print(o1, o2)
                plt.plot(p[1], p[0], 'ro')
                return


def perimeter(r):
    rd0 = circle(0, 0, r)
    rd1 = circle(0, 0, r + 1)
    rd0 = np.transpose(rd0, (1, 0))
    rd1 = np.transpose(rd1, (1, 0))
    rd0 = [tuple(o) for o in rd0]
    rd1 = [tuple(o) for o in rd1]
    rd = set(rd1) - set(rd0)
    rd = list(rd)
    rd = sorted(rd, key=lambda t: t[0] ** 2 + t[1] ** 2)
    return np.array(rd)


def surround():
    neiborhood = np.ones((3, 3))
    img_e = cv2.dilate(img, neiborhood, iterations=1)
    img_e = np.where((img_e == 1) & (img == 0), 1, 0)
    return img_e


def perimeter(r):
    rd = circle_perimeter(0, 0, r, method='andres')
    rd = np.transpose(rd, (1, 0))
    l = []
    for o in rd:
        l.append(tuple(o))
    return set(l)

rds = [perimeter(n) for n in range(1, 5)]
rd = rds[0]

def is_sandwiched(r, c):
    f = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    for n in range(4):
        a, b = f[n], f[(n + 4) % 8]
        if img[r + a[0], c + a[1]] == 0 and img[r + b[0], c + b[1]] == 0:
            return True
    return False

for r in range(28):
    for c in range(28):
        if img[r, c] == 0:
            continue
        if is_sandwiched(r, c):
            plt.plot(c, r, 'ro')

for con in contours:
    con = np.transpose(con, (1, 0))
    plt.plot(con[0], con[1], 'ro')



neiborhood4 = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]], np.uint8)
neiborhood8 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]])
img_e = cv2.erode(img, neiborhood4, iterations=1)


def count_dir(r, c):
    cc = np.array([0, 0])
    for o in rd:
        if img[r + o[0], c + o[1]] == 0:
            cc += np.array(o)
    return cc

dx = np.zeros((28, 28))
dy = np.copy(dx)

for r in range(28):
    for c in range(28):
        if img[r, c] == 0:
            continue
        v = count_dir(r, c)
        dx[r, c] = v[0]
        dy[r, c] = v[1]
pw = np.sqrt(dx ** 2 + dy ** 2)
pw = np.where(pw == 0, 1, pw)
dx /= pw; dy /= pw
plot_quiver(-dy, -dx)

def count_around(r, c):
    count = 0
    for o in rd:
        if img[r + o[0], c + o[1]] == 0:
            count += 1
    return count


for c1 in contours:
    for o in c1:
        img[o[1], o[0]] = 2
def peripheral(r, c):
    offset = [[1, 0], [-1, 0], [0, 1], [0, -1]]#, [1, 1], [-1, -1], [1, -1], [-1, 1]]
    return [(r + o[0], c + o[1]) for o in offset]
for c1 in contours:
    for n, o in enumerate(c1):
        o1 = clist.get(c1, n - 1, n + 2)
        o1 = [(o[1], o[0]) for o in o1]
        o2 = []
        for o in peripheral(o1[1][0], o1[1][1]):
            if img[o[0], o[1]] == 2:
                o2.append((o[0], o[1]))
        pair = list(set(o2) - set(o1))
        for o in pair:
            img[o[0], o[1]] = 3



contours = [np.transpose(o, (1, 0, 2))[0] for o in contours]
img = np.array(np.flip(img, axis=0))
plt.gca().invert_yaxis()


def perimeter(r):
    rd = circle_perimeter(0, 0, r, method='andres')
    rd = np.transpose(rd, (1, 0))
    l = []
    for o in rd:
        l.append(tuple(o))
    return set(l)

rd = perimeter(1)

for r in range(28):
    for c in range(28):
        if img[r, c] == 0:
            continue
        flag = True
        for o in rd:
            if img[r + o[0], c + o[1]] == 0:
                flag = False
        plt.plot([c], [r], 'bo')
        if flag:
            plt.plot([c], [r], 'ro')


img = img#[300: 330, 190: 220]
img = img#[50: 80, 205: 235]
img = img#[:,:,0]

dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
dy = -cv2.Sobel(img, cv2.CV_64F, 0, 1)
pw = np.sqrt(dx ** 2 + dy ** 2)
plt.imshow(img, cmap='gray')
plt.show()
pw = np.where(pw == 0, 1, pw)
dx /= pw; dy /= pw;
pw_b = np.where(pw > 20, 0, 1)

def quant_th(tht):
    thq = np.copy(tht)
    for i in range(-3, 4):
        thq[(np.pi*(2*i-1)/8 <= tht) & (tht < np.pi*(2*i+1)/8)] = i
    thq[(tht >= np.pi*7/8) | (tht < -np.pi*7/8)] = -4
    return thq

tht = np.arctan2(dx, -dy)#(-dy, dx)
thq = quant_th(tht)
plt.imshow(thq)
plt.show()
ver_f = np.array([[1], [-1]])
veq = cv2.filter2D(thq, -1, ver_f)
veq = np.where((veq != 0) | ((thq != 2) & (thq != -2)), 1, 0)
veq = np.where((veq != 0) | (pw_b != 0), 1, 0)

hor_f = np.array([[1, -1]])
heq = cv2.filter2D(thq, -1, hor_f)
heq = np.where((heq != 0) | ((thq != 0) & (thq != -4)), 1, 0)
heq = np.where((heq != 0) | (pw_b != 0), 1, 0)
ver_c = np.array([[1], [1], [1], [1], [1], [1]])
veq = np.where(veq == 0, 0, 1).astype(np.uint8)
veq = cv2.filter2D(veq, -1, ver_c)
veq = np.where(veq == 0, 0, 1).astype(np.uint8)
hor_f = np.array([[1, -1]])
heq = cv2.filter2D(thq, -1, hor_f)
heq = np.where(heq == 0, 0, 1)


plt.subplot(1, 2, 1)
plt.imshow(np.zeros((30, 30)))
tmp = np.where(dx == dx)
X, Y, U, V = tmp[1], tmp[0], dy, dx
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
plt.subplot(1, 2, 2)

plt.imshow(img, cmap='gray')
plt.show()


def grad_plot():
    for r in range(1, r_s):
        for c in range(c_s):
            a = img_f[r - 1, c] - img_f[r, c]
            b = np.linalg.norm(a)
            if b > 40:
                plt.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], 'r', lw = 1)
    for r in range(r_s):
        for c in range(1, c_s):
            a = img_f[r, c - 1] - img_f[r, c]
            b = np.linalg.norm(a)
            if b > 40:
                plt.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], 'r', lw = 1)


def patch_l():
    diff = np.array([[0, -1], [-1, 0], [1, 0], [0, 1]])
    def is_in(r, c):
        if (0 <= r and r < r_s) and (0 <= c and c < c_s):
            return True
        else:
            return False
    def dir_product():
        for r in range(r_s):
            for c in range(c_s):
                yield [r, c]
    for r, c in dir_product():
        total = np.zeros(2, np.int)
        total_c = 0
        for d in diff:
            nr, nc = r + d[0], c + d[1]
            if not is_in(nr, nc):
                total += d
                total_c += 1
                continue
            a = img_f[r, c] - img_f[nr, nc]
            b = np.linalg.norm(a)
            if b < 40:
                continue
            total += d
            total_c += 1
        if total_c < 2:
            continue
        if np.all(total == 0) or total_c > 2:
            label[r, c] = 1

import sys
sys.path.append('/home/hayato/machine_learning/Main/cpp')
import cr
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
img = np.array(np.flip(img, axis=0))
image, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
def smooth(route, sigma):
    rt = []
    size = len(route)
    if len(route) % 2 == 0:
        size -= 1
    e = int(size / 2)
    gs = cv2.getGaussianKernel(size, sigma, cv2.CV_64FC1)
    for n in range(len(route)):
        ps = clist.get(route, n - e, n + e + 1)
        ps = [p * g for p, g in zip(ps, gs)]
        p = np.sum(ps, axis=0)
        rt.append(p)
    return rt
mix_p = []
for i, route in enumerate(contours):
    route = route[:,0,:]
    rt = smooth(route, 2)
    for n in range(len(rt)):
        a = clist.get(rt, n, n + 2)
        mix_p.append(a)
ret_i, ret_p = cr.vec(mix_p)
for o in ret_i:
    n1, n2 = o
    node1 = mix_p[n1]
    m1 = (node1[0] + node1[1]) / 2
    node2 = mix_p[n2]
    m2 = (node2[0] + node2[1]) / 2
    plt.plot([m1[0], m2[0]], [m1[1], m2[1]], 'b')
for o in ret_p:
    p1, p2 = o
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b')
for o in mix_p:
    plt.plot([o[0][0], o[1][0]], [o[0][1], o[1][1]], 'ro', markersize=3)



def hist_plot():
    color = ('b','g','r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [64], [0, 256])
        plt.plot(histr,color = col)

def patch_l():
    diff = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
    def is_in(r, c):
        if (0 <= r and r < r_s) and (0 <= c and c < c_s):
            return True
        else:
            return False
    position = []
    r, c = 15, 15
    while True:
        dyed = False
        for d in diff:
            nr, nc = r + d[0], c + d[1]
            if not is_in(nr, nc):
                continue
            if label[nr, nc] != 0:
                continue
            a = img_f[r, c] - img_f[nr, nc]
            b = np.linalg.norm(a)
            if b > 40:
                continue
            label[nr, nc] = 1
            position.append([r, c])
            r, c = nr, nc
            dyed = True
        if len(position) == 0:
            break
        if not dyed:
            r, c = position.pop()


class Prepare:
    def __init__(self, num, new=False):
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)
        img = x_train[num]
        self.img_o = img


        img = prep2(img)
        img = prep3(img)
        self.img = np.copy(img)
        img = np.where(img == 2, 1, img)
        self.route = make_outline(img)


def prep(img):
    img = np.where(img < 128, 0, 1)
    img = img.reshape(28, 28)
    img = np.array(img, np.uint8)
    return img


def prep2(img):
    result = np.copy(img)
    filt = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]])
    for r in range(1, 27):
        for c in range(1, 27):
            if img[r, c] == 2:
                continue
            if np.tensordot(filt, img[r - 1: r + 2, c - 1: c + 2]) > 0:
                result[r, c] = 1
    return result
def prep3(img):
    result = np.copy(img)
    zero = np.where(img == 0, 1, 0)
    filt = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]])
    for r in range(1, 27):
        for c in range(1, 27):
            if img[r, c] != 1:
                continue
            if np.tensordot(filt, zero[r - 1: r + 2, c - 1: c + 2]) == 0:
                result[r, c] = 2
    return result
def prep4(img):
    filt = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]])
    result = np.copy(img)
    for r in range(1, 27):
        for c in range(1, 27):
            if img[r, c] == 2:
                continue
            if np.tensordot(filt, img[r - 1: r + 2, c - 1: c + 2]) >= 6:
                result[r, c] = 2
    img = np.copy(result)
    for r in range(1, 27):
        for c in range(1, 27):
            if img[r, c] == 0:
                continue
            if np.tensordot(filt, img[r - 1: r + 2, c - 1: c + 2]) <= 2:
                result[r, c] = 0
    return result
def make_outline(img):
    l = []
    for r in range(1, 27):
        for c in range(1, 27):
            if img[r, c] == 0:
                continue
            if img[r + 1][c] == 0 or img[r - 1][c] == 0 or img[r][c + 1] == 0 or img[r][c - 1] == 0:
                    l.append([r, c])
    rot = lambda t: t if t < 8 else rot(t - 8)
    cod = [(+1, -1), (+1, 0), (+1, +1), (0, +1), (-1, +1), (-1, 0), (-1, -1), (0, -1)]
    vnew = 0
    route = []
    while len(l) > 0:
        r, c = l[0][0], l[0][1]
        del l[0]
        route.append([])
        for n in range(8):
            z = cod[n]
            if img[r + z[0]][c + z[1]] == 0:
                vnew = n + 1
                break
        while True:
            if len(route[-1]) > 2:
                if route[-1][0] == route[-1][-2] and route[-1][1] == route[-1][-1]:
                    del route[-1][-2:]
                    break
            for n in range(7):
                z = cod[rot(vnew + n)]
                if img[r + z[0]][c + z[1]] == 1:
                    r += z[0]
                    c += z[1]
                    route[-1].append([r, c])
                    vnew = (rot(vnew + n) + 6) % 8
                    if [r, c] in l:
                        l.remove([r, c])
                    break
    return route


import numpy as np
cimport numpy as np
import mathlib as ml
import matplotlib.pyplot as plt

cdef class line:
    def __init__(self, **kwargs):
        self.p0 = kwargs['p0']
        self.t = kwargs['t']
        if 'p0' in kwargs and 'p1' in kwargs:
            self.p1 = kwargs['p1']
            v = self.p1 - self.p0
            self.theta = np.angle(complex(v[0], v[1]))
            if self.t == 2:
                self.k = np.linalg.norm(v)
        else:
            self.theta = kwargs['theta']
    cdef np.ndarray[np.float64_t, ndim=2] get_A(line self, line l):
        cdef np.ndarray[np.float64_t, ndim=2] o
        cdef np.ndarray[np.float64_t, ndim=0] a, b, c, d
        a = np.cos(self.theta)
        b = -np.cos(l.theta)
        c = np.sin(self.theta)
        d = -np.sin(l.theta)
        o = np.array([[a, b], [c, d]])
        return o
    def get_b(self, l):
        o = l.p0 - self.p0
        return o
    def in_range(self, k):
        if self.t == 0:
            return True
        elif self.t == 1:
            if k >= 0:
                return True
            else:
                return False
        else:
            if 0 <= k and k <= self.k:
                return True
            else:
                return False
    def point(self, k):
        return self.p0 + k * np.array([np.cos(self.theta), np.sin(self.theta)])
    def __its(self, l):
        cdef np.ndarray[np.float64_t, ndim=2] A = self.get_A(l)
        b = self.get_b(l)
        return np.linalg.solve(A, b)
    def its(self, l):
        k = self.__its(l)
        if self.in_range(k[0]) and l.in_range(k[1]):
            return self.point(k[0])
        else:
            raise Exception

def vect(o):
    return o[1] - o[0]
def slope(v):
    return np.angle(complex(v[0], v[1]))
def nrm_s(s, a):
    v1 = vect(s)
    v2 = [np.cos(a - np.pi / 2), np.sin(a - np.pi / 2)]
    if np.cross(v1, v2) < 0:
        return a - np.pi / 2
    else:
        return a + np.pi / 2
def vec(nodes):
    cdef int n1, n2, n3
    for n1 in range(len(nodes)):
        for n2 in range(n1 + 1, len(nodes)):
            seg = [nodes[n1], nodes[n2]]
            seg_c = [line(p0=seg[0][0], p1=seg[1][0], t=2), line(p0=seg[0][1], p1=seg[1][1], t=2)]
            try:
                seg_c[0].its(seg_c[1])
            except:
                continue
            seg_l = [line(p0=s[0], p1=s[1], t=2) for s in seg]
            v = [vect(o) for o in seg]
            s = slope(v[0])
            x = ml.interior_angle([-v[0], np.array([0, 0]), v[1]])
            if abs(x - 180) < 30:
                continue
            x = np.pi * x / 180
            avg_s = s + x / 2
            ray = []
            for s in seg:
                ns = nrm_s(s, avg_s)
                ray.append([line(p0=o, theta=ns, t=1) for o in s])
            cross_cnt = 0
            cross_p = np.zeros(2)
            for i in [[0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]:
                try:
                    r = ray[i[1]][i[2]]
                    s = seg_l[i[0]]
                    p = r.its(s)
                    cross_cnt += 1
                    if i[0] == 0:
                        cross_p += p
                    else:
                        cross_p += r.p0
                except:
                    continue
            if cross_cnt != 2:
                continue
            cross_p /= 2.0
            try:
                p = seg_l[1].its(line(p0=cross_p, theta=ray[0][0].theta, t=1))
            except:
                continue
            main_seg = line(p0=cross_p, p1=p, t=2)
            
            most_close = True
            for n3 in range(len(nodes)):
                if n3 == n1 or n3 == n2:
                    continue
                seg_l = line(p0=nodes[n3][0], p1=nodes[n3][1], t=2)
                try:
                    main_seg.its(seg_l)
                    most_close = False
                    break
                except:
                    continue
            if not most_close:
                continue
            r = main_seg
            p1 = r.p0; p2 = r.point(r.k)
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r')


frame_size = 6; frame_slide = 3
width, height = 28, 28
partition = []
for wc in np.arange((width / 2) % frame_size, width + 1, frame_slide):
    for hc in np.arange((height / 2) % frame_size, height + 1, frame_slide):
        partition.append([[wc - frame_size / 2, hc - frame_size / 2], []])
def in_square(p, lb, size):
    left, bot = lb
    cw = left < p[0] and p[0] < left + size
    ch = bot < p[1] and p[1] < bot + size
    if cw and ch:
        return True
    else:
        return False

for i, mp in enumerate(mix_p):
    for pn in partition:
        lb = pn[0]
        if np.any([in_square(mp[n], lb, frame_size) for n in range(2)]):
            pn[1].append(i)
for o in partition:#48, 49
    l, b = o[0]; s = frame_size
    plt.plot([l, l + s, l + s, l, l], [b, b, b + s, b + s, b], 'b')
    for o1 in o[1]:
        o1 = mix_p[o1]
        plt.plot([o1[0][0], o1[1][0]], [o1[0][1], o1[1][1]], 'ro', markersize=3)
    vec([mix_p[n] for n in o[1]])


def lineA(p1, p2):
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0] * p2[1] - p2[0] * p1[1]
    return np.array([a, b, c])
def lineB(p1, p2):
    o = p2 - p1
    return [p1, np.angle(np.complex(o[0], o[1]))]
def intersectionA(l1, l2):
    A = np.array([
        [l1[0], l1[1]],
        [l2[0], l2[1]]])
    b = -np.array([l1[2], l2[2]])
    return np.linalg.solve(A, b)
def intersectionB(l1, l2):
    A = np.array([
        [np.cos(l1[1]), -np.cos(l2[1])],
        [np.sin(l1[1]), -np.sin(l2[1])]])
    b = np.array([l2[0][0] - l1[0][0], l2[0][1] - l1[0][1]])
    return np.linalg.solve(A, b)
def intersectionC(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy):
    cdef:
        double ta, tb, tc, td
    ta = (cx - dx) * (ay - cy) + (cy - dy) * (cx - ax);
    tb = (cx - dx) * (by - cy) + (cy - dy) * (cx - bx);
    tc = (ax - bx) * (cy - ay) + (ay - by) * (ax - cx);
    td = (ax - bx) * (dy - ay) + (ay - by) * (ax - dx);
    return tc * td < 0 and ta * tb < 0

p1, p2 = ray[1].p1, ray[1].p2
            ray_p = [sp.Float(o) for o in [p1.x, p1.y, p2.x, p2.y]]
            p1, p2 = segment[0].p1, segment[0].p2
            seg_p = [sp.Float(o) for o in [p1.x, p1.y, p2.x, p2.y]]
            plt.plot([ray_p[0], ray_p[2]], [ray_p[1], ray_p[3]])
            plt.plot([seg_p[0], seg_p[2]], [seg_p[1], seg_p[3]], 'bo')

p1 = route[n1]; p2 = route[n2]
            a = ml.interior_angle([p1[0], p1[1], p2[1]])
            b = ml.interior_angle([p1[1], p1[0], p2[0]])
            if a > 90 and b > 90:
                continue
            if a > 180:
                a = 360 - a
            if b > 180:
                b = 360 - b
            if a < 90 and b < 90:
                print(n1, n2, a, b)
                o1 = p1; o2 = p2;
                p1 = o1[0]; p2 = o1[1]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'bo')
                p1 = o2[0]; p2 = o2[1]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'bo')


def vec(route):
    l = []
    for a in route:
        m = (a[0] + a[1]) / 2
        t = a[1] - a[0]
        n = np.array([t[1], -t[0]])
        base_line = ml.line(m, m + n)
        candidate = []
        for b in route:
            if np.all(np.array(a) == np.array(b)):
                continue
            tgt_line = ml.line(b[0], b[1])
            p3 = ml.intersection(base_line, tgt_line)
            if np.all((b[0] - p3) * (b[1] - p3) > 0):
                continue
            if np.cross(t, p3 - m) > 0:
                continue
            angle = abs(ml.external_angle([m, p3, b[0]]))
            candidate.append([p3, np.linalg.norm(m - p3), angle])
        candidate = sorted(candidate, key=lambda t: t[1])
        if candidate[0][2] < 60 or 120 < candidate[0][2]:
            continue
        p = candidate[0][0]
        #p = (m + p) / 2.0
        plt.plot([m[0], p[0]],[m[1], p[1]], 'r', markersize=2)
        #plt.plot(p[0], p[1], 'ro', markersize=2)
    return l


    length = cv2.arcLength(route, True)
    circle_area = (length ** 2) / (4 * np.pi)
    area = cv2.contourArea(route)
    rate = area / circle_area
    print(rate)

def detect_contour(path):
    src = cv2.imread(path, 0)
    gray = src#cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    image, contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    detect_count = 0
    
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        
        if area < 1e2 or 1e5 < area:
            continue
        
        if len(contours[i]) > 0:
            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        detect_count = detect_count + 1
    plt.imshow(src, cmap='gray')
    plt.show()

if __name__ == '__main__':
    detect_contour('/home/hayato/Downloads/IMG_20170913_102735.jpg')

def k_curv(l, k):
    xs = range(len(l))
    ys = []
    for n in xs:
        p = [clist.get(l, n - k), clist.get(l, n), clist.get(l, n + k)]
        ea = external_angle(p)
        ys.append(ea)
    return [xs, ys]
def detect_edge(ys):
    extreme = []
    for n in range(len(ys)):
        os = clist.get(ys, n - 1, n + 2)
        os -= os[1]
        if os[0] * os[2] <= 0:
            continue
        extreme.append(n)
    while True:
        removed = False
        for n in range(len(extreme)):
            il = clist.get(extreme, n, n + 4)
            lo = [ys[n] for n in il]
            ld = [abs(lo[n] - lo[n + 1]) for n in range(3)]
            if ld[1] < 5 and ld[1] < ld[0] and ld[1] < ld[2]:
                extreme.remove(il[1])
                extreme.remove(il[2])
                removed = True
                break
        if not removed:
            break
    return [n for n in extreme if (ys[n] > 10 and ys[n] > ys[n - 1]) or (ys[n] < -10 and ys[n] < ys[n - 1])]
def detect_flat(ys, t):
    l = []
    start = -1
    for n in range(len(ys) * 2):
        a = clist.get(ys, n - 1, n + 2)
        if abs(a[0]) > t and abs(a[1]) <= t:
            start = n
        elif abs(a[1]) <= t and abs(a[2]) > t:
            if start != -1:
                l.append([start % len(ys), n % len(ys)])
            if n >= len(ys):
                break
    return l
    xs, ys = k_curv(rt, 1)
    es = detect_edge(ys)
    #fs = detect_flat(ys, 5)
    
    plt.subplot(1, len(contours) + 1, i + 1)
    plt.plot(xs, ys)
    for n in es:
        plt.plot(xs[n], ys[n], 'ro')
    plt.plot(xs, [-5 for e in ys], 'b')
    plt.plot(xs, [5 for e in ys], 'b')
    for p in fs:
        plt.plot([xs[p[0]], xs[p[1]]], [ys[p[0]], ys[p[1]]], 'bo')
    for n in es:
        plt.plot(xs[n], ys[n], 'ro')
    for p in fs:
        plt.plot([xs[p[0]], xs[p[1]]], [ys[p[0]], ys[p[1]]], 'bo')
def p_search(l, plus):
    if plus:
        comp = lambda t: t > 0
    else:
        comp = lambda t: t < 0
    l2 = []
    for n in range(len(l)):
        ps_i = clist.get(l, n - 1, n + 2)
        ps = [route[n] for n in ps_i]
        if comp(external_angle(ps)):
            l2.append(l[n])
    return l2
lp = [n for n in range(len(route))]
lm = [n for n in range(len(route))]
lpm = [lp]; lmm = [lm]
for e in range(4):
    lp = p_search(lp, True)
    lm = p_search(lm, False)
    lpm.append(lp); lmm.append(lm)
print(lpm[1:])
sp = []
for n in range(1, len(lpm) - 1):
    a = [o for o in lpm[n] if not o in lpm[n + 1]]
    for o in a:
        i1 = lpm[n - 1].index(o)
        sp.append(lpm[n - 1][i1 - 1: i1 + 2])
print(sp)
for n, o in enumerate(route):
    plt.text(o[1], o[0], str(n))
for n in lp:
    plt.text(route[n][1] - 0.25, route[n][0] + 0.25, 'o')
for n in lm:
    plt.text(route[n][1] - 0.25, route[n][0] + 0.25, 'x')


def p_search(l, plus):
    if plus:
        comp = lambda t: t > 0
    else:
        comp = lambda t: t < 0
    l2 = []
    for n in range(len(l)):
        ps_i = clist.get(l, n - 1, n + 2)
        ps_m = [o[1] for o in ps_i]
        ps = [route[n] for n in ps_m]
        if comp(external_angle(ps)):
            l2.append([ps_i[0][0], ps_i[1][1], ps_i[2][2]])
    return l2
lp = [[n, n, n] for n in range(len(route))]
lm = [[n, n, n] for n in range(len(route))]
for e in range(1):
    lp = p_search(lp, True)
    lm = p_search(lm, False)

for n in lp:
    n = n[1]
    plt.text(route[n][1] - 0.25, route[n][0] + 0.25, 'o')
for n in lm:
    n = n[1]
    plt.text(route[n][1] - 0.25, route[n][0] + 0.25, 'x')


def smooth(route):
    rt = []
    for n in range(len(route)):
        ps = clist.get(route, n - 2, n + 3)
        gs = cv2.getGaussianKernel(5, 1, cv2.CV_64FC1)
        ps = [p * g for p, g in zip(ps, gs)]
        p = np.sum(ps, axis=0)
        rt.append(p)
    return rt
rt = route
for e in range(1):
    rt = smooth(rt)
plt.imshow(img)
xs = [o[1] for o in rt]; xs.append(xs[0])
ys = [o[0] for o in rt]; ys.append(ys[0])
plt.plot(xs, ys, color='r')
plt.show()

def internal_angle(p):
    p -= p[0]
    ff = complex(p[2][0], p[2][1]) / complex(p[1][0], p[1][1])
    return np.angle(ff) * 180 / np.pi
def route_available(num):
    l = clist.range(route, num - 1)
    ia = 0
    for n in range(num - 2, num - len(route), -1):
        a = external_angle([clist.get(route, num), clist.get(route, l[-1]), clist.get(route, n)])
        ia += internal_angle([clist.get(route, num), clist.get(route, n + 1), clist.get(route, n)])
        if a < 0 and ia < 0:
            l.extend(clist.range(route, n))
    return l

av_left = []
for n in range(len(route) - 1, -1, -1):
    av_left.append(route_available(n))
    print(n, av_left[-1])

av_right = [[] for e in av_left]
for n1, o in enumerate(av_left):
    for n2 in o:
        av_right[n2].append(n1)

lt = [[] for e in av_left]
for n1, o1 in enumerate(av_left):
    for n2 in o1:
        flag = False
        for n3 in av_left[n2]:
            a = external_angle([route[n1], route[n2], route[n3]])
            if a >= 0:
                flag = True
        if flag:
            lt[n1].append(n2)
lt2 = [[] for e in av_left]
for n2, o2 in enumerate(lt):
    for n3 in o2:
        flag = False
        for n1 in av_right[n2]:
            a = external_angle([route[n1], route[n2], route[n3]])
            if a >= 0:
                flag = True
        if flag:
            lt2[n2].append(n3)
curv_p3 = []
for n1, o1 in enumerate(lt2):
    for n2 in o1:
        for n3 in lt2[n2]:
            if external_angle([route[n1], route[n2], route[n3]]) >= 0:
                curv_p3.append([n1, n2, n3])
for o in curv_p3:
    pass#print(o)

fig, ax = plt.subplots()
ax.imshow(img)
for n in range(len(route)):
    ax.text(route[n][1] - 0.25, route[n][0] + 0.25, str(n))
for o in curv_p3:
    line = plt.Line2D([route[n][1] for n in o], [route[n][0] for n in o], color='r')
    ax.add_line(line)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.axis('off')
plt.show()

def route_available(num):
    l = clist.range(route, num + 1)
    ia = 0
    for n in range(num + 2, num + int(len(route) / 2)):
        a = external_angle([clist.get(route, num), clist.get(route, l[-1]), clist.get(route, n)])
        ia += internal_angle([clist.get(route, num), clist.get(route, n - 1), clist.get(route, n)])
        if a < 0 and ia < 0:
            l.extend(clist.range(route, n))
    return l

def cycle_rec(l, st, n):
    st.append(n)
    if n > 60:
        return
    if len(st) >= 3:
        p = [route[st[-n]] for n in range(3, 0, -1)]
        if external_angle(p) < 0:
            st.pop()
            return
    cycle_rec(l, st, l[n][0])

lt2 = [[] for e in l]
for n2, o2 in enumerate(lt):
    for n3 in o2:
        flag = False
        for n1 in lr[n2]:
            a = external_angle([route[n1], route[n2], route[n3]])
            if a >= 0:
                flag = True
        if flag:
            lt2[n2].append(n3)

lt = [[] for e in l]
for n1, o1 in enumerate(l):
    for n2 in o1:
        flag = False
        for n3 in l[n2]:
            a = external_angle([route[n1], route[n2], route[n3]])
            if a >= 0:
                flag = True
        if flag:
            lt[n1].append(n2)

def line_eq(p1, p2):
    return [p1[1] - p2[1], p2[0] - p1[0], p1[0] * p2[1] - p2[0] * p1[1]]
def ver_distance(a, b, c, p):
    return (a * p[0] + b * p[1] + c) / np.sqrt(a ** 2 + b ** 2)
def convex_hull(points):
    index = range(len(points))
    index = sorted(index, key=lambda i: (points[i][0], points[i][1]))
    if len(points) <= 1:
        return index
    def cross(o, a, b):
        return (points[a][0] - points[o][0]) * (points[b][1] - points[o][1]) - (points[a][1] - points[o][1]) * (points[b][0] - points[o][0])
    lower = []
    for n in index:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], n) <= 0:
            lower.pop()
        lower.append(n)
    upper = []
    for n in reversed(index):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], n) <= 0:
            upper.pop()
        upper.append(n)
    return lower[:-1] + upper[:-1]

ch = convex_hull(route)

class Curve:
    def __init__(self):
        self.l = []
l = []
for n in range(len(route)):
    c = Curve()
    c.l.extend(clist.range(route, n, n + 2))
    l.append(c)
for c in l:
    print(c.l)


data = np.ones((20, len(route))) * np.inf

fig, ax = plt.subplots()

for start in range(len(route)):
    for n in range(20):
        if n == 0:
            data[n][start] = 0
            continue
        s = clist.get(route, start, start + n + 2)
        a = s[0][1] - s[-1][1]
        b = s[-1][0] - s[0][0]
        c = s[0][0] * s[-1][1] - s[-1][0] * s[0][1]
        l = [(a * o[0] + b * o[1] + c) / np.sqrt(a ** 2 + b ** 2) for o in s[1: -1]]
        l = np.array(l)
        #if np.all(l > 0) or np.all(l < 0):
        #    continue
        data[n][start] = np.sum(np.abs(l)) / (n + 2)
line_i = []
for start in range(len(route)):
    for n in range(19):
        aso = np.ones((3, 3)) * np.inf
        da = np.array([
            [start - 1, n],
            [start, n - 1],
            [(start + 1) % len(route), n - 2]])
        for n1 in range(3):
            for n2 in range(3):
                if da[n1][1] + n2 < 0 or da[n1][1] + n2 >= 20:
                    continue
                aso[n2][n1] = data[da[n1][1] + n2, da[n1][0]]
        if np.min(aso) == np.inf:
            continue
        aw = np.argwhere(aso == np.min(aso))
        cont = False
        for o in aw:
            if np.all(o == [1, 0]) or np.all(o == [2, 1]) or np.all(o == [2, 0]):
                cont = True
                break
        if cont:
            continue
        if np.min(aso) == aso[1, 1]:
            #line_i.append([start, start + n + 1])
            #ax.text(start, n, 'o')
            pass

for o in line_i:
    p1 = clist.get(route, o[0])
    p2 = clist.get(route, o[1])
    line = plt.Line2D([p1[1], p2[1]], [p1[0], p2[0]], color='r')
    ax.add_line(line)
for n in range(len(route)):
    p = route[n]
    ax.text(p[1] - 0.25, p[0] + 0.25, 'o')
ax.imshow(img)
ax.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.show()
#38, 11 - 51, 8


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

r = []
for o in route:
    r.append(tuple(o))
ch = convex_hull(r)


a = r.index(ch[3])
b = r.index(ch[4])
ch = convex_hull(r[a: b + 1])
a = r.index(ch[2])
b = r.index(ch[1])
ch = convex_hull(r[a: b + 1])

plt.imshow(img)
for n, o in enumerate(ch):
    plt.text(o[1], o[0], str(n))
plt.show()

a = [0]
edge = []
while True:
    l = [(n, np.linalg.norm(o - route[a[-1]])) for n, o in enumerate(route)]
    a.append(sorted(l, key=lambda t: t[1])[-1][0])
    if len(a) < 3:
        continue
    if a[-1] == a[-3]:
        edge.extend([a[-2], a[-1]])
        break
b = clist.range(route, edge[0], edge[1] + 1)
c = clist.range(route, edge[1], edge[0] + 1)
plt.imshow(img)
plt.show()


l = [[[0], 0]]
for n in range(1, len(route)):
    direction = np.array([
        [0, 2, 0],
        [3, 0, 1],
        [0, 4, 0]])
    prev = np.zeros((3, 3))
    prev[route[n - 1][0] - route[n][0] + 1, route[n - 1][1] - route[n][1] + 1] = 1
    d = np.tensordot(prev, direction)
    if d > 0:
        l[-1][0].append(n)
        l[-1][1] = d
    else:
        l.append([[n], 0])

def is_triangle(n):
    o = clist.get(l, n, n + 3)
    top = route[o[0][0][0]]
    mid = route[o[1][0][0]]
    bot = route[o[2][0][0]]
    if np.any(((top - mid) * (bot - mid)) > 0):
        return True
    else:
        return False

lt = [n + 1 for n in range(len(l)) if is_triangle(n)]
def split(l, sep):
    lt = [[]]
    for n in range(len(l)):
        if sep(l[n]):
            lt.append([])
        else:
            lt[-1].append(l[n])
    return [o for o in lt if o != []]
st = []
for n in range(len(lt)):
    ist = clist.get(lt, n, n + 2)
    if clist.range(l, ist[0] + 1) == ist[1]:
        continue
    il = clist.range(l, ist[0], ist[1] + 1)
    sup = max([len(l[n][0]) for n in il])
    for n1 in range(sup, 1, -1):
        ilst = split(il, lambda i: len(l[i][0]) != n1 - 1 and len(l[i][0]) != n1)
        for o in ilst:
            st.append(o)
st = sorted(st, key=lambda t: len(t))
stt = []
for n1 in range(len(st)):
    leave = True
    for n2 in range(n1 + 1, len(st)):
        if set(st[n1]) <= set(st[n2]):
            leave = False
            break
    if leave:
        stt.append(st[n1])
st = sorted(stt, key=lambda t: t[0])
print(st)

fig, ax = plt.subplots()
ax.imshow(img)

for o in st:
    a1 = route[l[o[0]][0][0]]
    a2 = route[l[o[-1]][0][-1]]
    line = plt.Line2D([a1[1], a2[1]], [a1[0], a2[0]], color='r')
    ax.add_line(line)

plt.imshow(img)
plt.show()



lc = [o[1] - o[0] + 1 for o in l]
lc = [n for n, o in enumerate(lc) if o == 1 or o == 2]
start = 0
for n in range(len(lc) - 1):
    if lc[n + 1] - lc[n] != 1:
        start = n + 1
        break
lt = [[lc[start], lc[start]]]
for n in range(start, start + len(lc)):
    o1 = lc[n % len(lc)]
    if n == len(lc) - 1:
        o2 = -1
    else:
        o2 = lc[(n + 1) % len(lc)]
    if o2 - o1 == 1:
        lt[-1][1] = lc[(n + 1) % len(lc)]
    else:
        lt.append([lc[(n + 1) % len(lc)], lc[(n + 1) % len(lc)]])
lt = [o for o in lt if o[0] != o[1]]


start = 0
for n in range(-2, len(l) - 2):
    if is_triangle(n):
        start = n
        break
lt = [[start + 1, start + 1]]
for n in range(start + 1, start + len(l)):
    if is_triangle(n):
        lt.append([n + 1, n + 1])
    else:
        lt[-1][1] = n + 2
print(lt)

l = [[0, 0, 0]]
for n in range(1, len(route)):
    direction = np.array([
        [0, 2, 0],
        [3, 0, 1],
        [0, 4, 0]])
    prev = np.zeros((3, 3))
    prev[route[n - 1][0] - route[n][0] + 1, route[n - 1][1] - route[n][1] + 1] = 1
    d = np.tensordot(prev, direction)
    if d > 0:
        l[-1][1] = n
        l[-1][2] = d
    else:
        l.append([n, n, 0])
lc = [0 for e in l]

def external_angle(p):
    p -= p[1]
    ff = -complex(p[2][0], p[2][1]) / complex(p[0][0], p[0][1])
    return np.angle(ff) * 180 / np.pi
for n in range(-1, len(l) - 1):
    tn = l[n][1] - l[n][0] + 1
    bn = l[n + 1][1] - l[n + 1][0] + 1
    if tn == bn:
        if tn == 1 or l[n][2] == l[n + 1][2]:
            lc[n] = 3
            continue
    if tn >= bn:
        ea = external_angle([
            np.array(route[l[n][1] - 1]),
            np.array(route[l[n][1]]),
            np.array(route[l[n + 1][0]])])
    else:
        ea = external_angle([
            np.array(route[l[n][1]]),
            np.array(route[l[n + 1][0]]),
            np.array(route[l[n + 1][0] + 1])])
    if ea > 0:
        lc[n] = 1
    else:
        lc[n] = 2
lp = [False for e in l]
lo = []
def spread(p, d, t):
    p += d; pa = p % len(l)
    if lc[pa] == 3 or lc[pa] == t:
        lp[pa] = True
        if d > 0:
            lo[-1].append(p + d)
        else:
            lo[-1].insert(0, p)
        spread(p, d, t)
for n, o in enumerate(lc):
    if o == 3:
        continue
    if lp[n]:
        continue
    lo.append([n, n + 1])
    lp[n] = True
    spread(n, 1, o)
    spread(n, -1, o)


for n in range(-2, len(l) - 2):
    top = np.array(route[l[n][1]])
    mid = (np.array(route[l[n + 1][0]]) + np.array(route[l[n + 1][1]])) / 2
    bot = np.array(route[l[n + 2][0]])
    if np.all(((top - mid) * (bot - mid)) < 0):
        continue
    if external_angle([top, mid, bot]) > 0:
        lc[n][0] = lc[n + 1][0] = lc[n + 2][0] = True
    else:
        lc[n][1] = lc[n + 1][1] = lc[n + 2][1] = True

        while True:
            sv = np.copy(img)
            img = prep4(img)
            img = prep2(img)
            img = prep3(img)
            tmp = np.where(img == 1, 0, img)
            if np.allclose(sv, tmp):
                break
            else:
                img = tmp
        f_i = '/home/hayato/machine_learning/data/i' + str(num) + '.pkl'
        f_r = '/home/hayato/machine_learning/data/r' + str(num) + '.pkl'
        if ((not new) and
            os.path.exists(f_i) and
            os.path.exists(f_r)):
            with open(f_i, mode='rb') as f:
                self.img = pickle.load(f)
            with open(f_r, mode='rb') as f:
                self.route = pickle.load(f)
            
            return

        with open(f_i, mode='wb') as f:
            pickle.dump(self.img, f)
        with open(f_r, mode='wb') as f:
            pickle.dump(self.route, f)

#fig, ax = plt.subplots()

def prep4(img):
    result = np.copy(img)
    for r in range(1, 27):
        for c in range(1, 27):
            if img[r, c] == 1:
                continue
            if ((img[r - 1, c] == 1 and img[r + 1, c] == 1) or
                (img[r, c - 1] == 1 and img[r, c + 1] == 1)):
                result[r, c] = 1
    img = np.copy(result)
    for r in range(1, 27):
        for c in range(1, 27):
            if img[r, c] == 0:
                continue
            if ((img[r - 1, c] == 0 and img[r + 1, c] == 0) or
                (img[r, c - 1] == 0 and img[r, c + 1] == 0)):
                result[r, c] = 0
    return result


p, q, r = (1, 1, 1)
for n in range(40):
    S = 0
    Sp, Sq, Sr = (0, 0, 0)
    for o in route[19: 32]:
        s = ((o[0] - p) ** 2 + (o[1] - q) ** 2 - r) ** 2
        S += s
        Sp += -2 * s * (o[0] - p) * (((o[0] - p) ** 2 + (o[1] - q) ** 2) ** -0.5)
        Sq += -2 * s * (o[1] - q) * (((o[0] - p) ** 2 + (o[1] - q) ** 2) ** -0.5)
        Sr += -2 * s
    p -= S / Sp
    q -= S / Sq
    r -= S / Sr
    print(p, q, r)


def middle_point(p1, p2):
    r1, c1 = p1
    r2, c2 = p2
    mr = (r1 + r2) / 2
    mc = (c1 + c2) / 2
    return (mr, mc)
def normal_vec(p1, p2):
    r1, c1 = p1
    r2, c2 = p2
    dr = (r1 - r2) / 2
    dc = (c1 - c2) / 2
    a = np.sqrt(dr ** 2 + dc ** 2)
    return (-dc, dr) / a
k = 14
"""
prev = (0, 0)
start = -1
stop = -1
roop = -1
l = []
for n in range(len(route) - 1):
    if n == roop:
        break
    vec = normal_vec(route[n], route[n + k])
    cur = np.angle(np.complex(vec[0], vec[1]))
    if n == 0:
        pass
    elif prev != cur:
        if stop == n - 1:
            if roop == -1:
                roop = start + int(len(route) / 2)
            l.append((start, stop + k))
        start = n
    else:
        if start != -1:
            stop = n
    prev = cur
"""
ax.imshow(img_s)
"""
for n in range(int(len(route) / 2)):
    vec = normal_vec(route[n], route[n + k])
    mp = middle_point(route[n], route[n + k])
    ll = 2
    line = matplotlib.lines.Line2D(
        [mp[1] - vec[1] * ll, mp[1] + vec[1] * ll],
        [mp[0] - vec[0] * ll, mp[0] + vec[0] * ll], color='r')
    ax.add_line(line)
"""
for n in range(int(len(route) / 2)):
    vec1 = normal_vec(route[n], route[n + k])
    vec2 = normal_vec(route[n + 1], route[n + k + 1])
    mp1 = middle_point(route[n], route[n + k])
    mp2 = middle_point(route[n + 1], route[n + k + 1])
    A = np.array([
        [vec1[1], -vec1[0]],
        [vec2[1], -vec2[0]]])
    b = np.array([
        vec1[1] * mp1[0] - vec1[0] * mp1[1],
        vec2[1] * mp2[0] - vec2[0] * mp2[1]])
    try:
        p = np.linalg.solve(A, b)
        if np.all(p >= 0) and np.all(p <= 27):
            circle = plt.Circle([p[1], p[0]], 0.2, color='r', fill=False)
            ax.add_artist(circle)
            """
            line = matplotlib.lines.Line2D(
                [mp1[1], p[1]],
                [mp1[0], p[0]], color='r')
            ax.add_line(line)
            line = matplotlib.lines.Line2D(
                [mp2[1], p[1]],
                [mp2[0], p[0]], color='r')
            ax.add_line(line)
            """
    except np.linalg.linalg.LinAlgError:
        pass


"""
    ll = 2
    line = matplotlib.lines.Line2D(
        [mp[1] - vec[1] * ll, mp[1] + vec[1] * ll],
        [mp[0] - vec[0] * ll, mp[0] + vec[0] * ll], color='r')
    ax.add_line(line)
"""
plt.show()







def curvature(route):
    route = np.array(route)
    a = route[0][1] - route[-1][1]
    b = route[-1][0] - route[0][0]
    c = route[0][0] * route[-1][1] - route[-1][0] * route[0][1]
    straight = True
    for p in route:
        if a * p[0] + b * p[1] + c != 0:
            straight = False
            break
    if straight:
        return 0
    p = route.T
    x, y = p
    A = np.array([
        [np.sum(x ** 2), np.sum(x * y), np.sum(x)],
        [np.sum(x * y), np.sum(y ** 2), np.sum(y)],
        [np.sum(x), np.sum(y), len(route)]])
    b = np.array([
        -np.sum(x ** 3 + x * y ** 2),
        -np.sum(y * x ** 2 + y ** 3),
        -np.sum(x ** 2 + y ** 2)])
    k = np.linalg.solve(A, b)
    p = -k[0] / 2
    q = -k[1] / 2
    r = (p ** 2 + q ** 2 - k[2]) ** 0.5
    R = r ** -1
    route = route - np.array([-k[0: 2] / 2 for o in route])
    a = route[0]
    a = np.angle(np.complex(a[0], a[1]))
    b = route[int(len(route) / 2)]
    b = np.angle(np.complex(b[0], b[1]))
    c = route[-1]
    c = np.angle(np.complex(c[0], c[1]))
    if a > c:
        if a > b and b > c:
            R *= -1
    else:
        if a >= b or b >= c:
            R *= -1
    return (R, p, q, r)



R = np.ones((28, 28)) * -1

for r in range(1, 27):
    for c in range(1, 27):
        if img[r, c] != 1:
            continue
        cut = img[r - 1: r + 2, c - 1: c + 2]
        p = np.argwhere(cut == 1)
        A = np.concatenate((p, [[1], [1], [1]]), axis=1)
        b = -np.sum(p ** 2, axis = 1)
        cur = 0
        try:
            x = np.linalg.solve(A, b)
            cur = (x[0] ** 2 / 4 + x[1] ** 2 / 4 - x[2]) ** -0.5
        except:
            pass
        R[r, c] = cur



def make_outline(img):
    l = []
    for r in range(1, 27):
        for c in range(1, 27):
            if img[r, c] == 0:
                continue
            if img[r + 1][c] == 0 or img[r - 1][c] == 0 or img[r][c + 1] == 0 or img[r][c - 1] == 0:
                    l.append([r, c])
    rot = lambda t: t if t < 8 else rot(t - 8)
    cod = [(+1, -1), (+1, 0), (+1, +1), (0, +1), (-1, +1), (-1, 0), (-1, -1), (0, -1)]
    vnew = 0
    route = []
    while len(l) > 0:
        r, c = l[0][0], l[0][1]
        del l[0]
        route.append([])
        for n in range(8):
            z = cod[n]
            if img[r + z[0]][c + z[1]] == 0:
                vnew = n + 1
                break
        while True:
            if len(route[-1]) > 2:
                if route[-1][0] == route[-1][-2] and route[-1][1] == route[-1][-1]:
                    del route[-1][-2:]
                    break
            for n in range(7):
                z = cod[rot(vnew + n)]
                if img[r + z[0]][c + z[1]] == 1:
                    r += z[0]
                    c += z[1]
                    route[-1].append([r, c])
                    vnew = (rot(vnew + n) + 6) % 8
                    if [r, c] in l:
                        l.remove([r, c])
                    break
    return route



fs = np.array([[
    [0, 0],
    [0, 0]],[
    
    [0, 1],
    [0, 0]],[
    
    [1, 1],
    [0, 0]],[
    
    [0, 1],
    [1, 0]],[
    
    [1, 1],
    [1, 0]],[
    
    [1, 1],
    [1, 1]]])

filt = np.array([
    [2, 1],
    [4, 8]])

def rotate(img):
    tmp = img[0, 1]
    img[0, 1] = img[0, 0]
    img[0, 0] = img[1, 0]
    img[1, 0] = img[1, 1]
    img[1, 1] = tmp

def sbp(img):
    width = img.shape[0]
    height = img.shape[1]
    result = np.zeros((width - 1, height - 1), np.uint8)
    for r in range(width - 1):
        for c in range(height - 1):
            i = img[r: r + 2, c: c + 2]
            m = 15
            mi = np.copy(i)
            for n in range(4):
                v = np.tensordot(filt, i)
                if v == 0 or v == 15:
                    break
                if v <= m:
                    m = v
                    mi = np.copy(i)
                rotate(i)
            for i, f in enumerate(fs):
                if np.all(f == mi):
                    result[r, c] = i
                    break
    return result

def rotate2(ex):
    ex = [ex[0, 1], ex[0, 0], ex[1, 0], ex[1, 1]]
    ex2 = [ex]
    for n in range(3):
        ex = np.append(ex[1:], ex[0])
        ex2.append(ex)
    for n in range(3, 0, -1):
        ex = np.amin(ex2, axis=0)
        ex2 = [o for o in ex2 if o[n] == ex[n]]
    ex = ex2[0]
    ex = [[ex[1], ex[0]], [ex[2], ex[3]]]
    return ex

def def_style(img):
    r = [sorted([img[0, 0], img[1, 1]]), sorted([img[1, 0], img[0, 1]])]
    return np.array(sorted(r))

img = sbp(img)
fs = [[] for n in range(17)]
fc = [[] for n in range(17)]
plt.imshow(img)
plt.show()

for r in range(26):
    for c in range(26):
        i = img[r: r + 2, c: c + 2]
        d = def_style(i)
        t = np.sum(i)
        if fs[t] is []:
            fs[t].append(d)
            fc[t].append(1)
            continue
        apnd = True
        for i, f in enumerate(fs[t]):
            if np.all(f == d):
                apnd = False
                fc[t][i] += 1
                break
        if apnd:
            fs[t].append(d)
            fc[t].append(1)
for c in range(17):
    for r in range(len(fc[c])):
        plt.subplot(3, 17, r * 17 + c + 1)
        f = fs[c][r]
        f = [[f[0, 0], f[1, 1]], [f[1, 0], f[0, 1]]]
        plt.imshow(f, vmin=0, vmax=4)
        plt.text(0, -1, fc[c][r])
        plt.axis('off')
plt.show()


r = sbp(img)
r = [np.sum(o) / (28 * 28) for o in r]

p = np.sum(img) / (28 * 28)
prob = []
prob.append((1 - p) ** 4 * p ** 0 * 1)
prob.append((1 - p) ** 3 * p ** 1 * 4)
prob.append((1 - p) ** 2 * p ** 2 * 4)
prob.append((1 - p) ** 2 * p ** 2 * 2)
prob.append((1 - p) ** 1 * p ** 3 * 4)
prob.append((1 - p) ** 0 * p ** 4 * 1)

plt.bar(range(6), np.array(r) / np.array(prob), width=0.4)
#plt.bar(np.arange(0.4, 6.4, 1.0), r, width=0.4)
plt.show()

for r in range(27):
    for c in range(27):
        i = img[r: r + 2, c: c + 2]
        m = 15
        mi = np.copy(i)
        for n in range(4):
            v = np.tensordot(filt, i)
            if v <= m:
                m = v
                mi = np.copy(i)
                if v == 0:
                    break
            rotate(i)
        if fs is []:
            fs.append(mi)
            fc.append(1)
            continue
        apnd = True
        for i, f in enumerate(fs):
            if np.all(f == mi):
                apnd = False
                fc[i] += 1
                break
        if apnd:
            fs.append(mi)
            fc.append(1)
