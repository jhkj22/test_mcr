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
