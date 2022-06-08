import numpy as np
from scipy.integrate import quad

eV = 1.6e-19
hbar = 1.05e-34
m = 0.067 * 9.1e-31
w = 5e-9
d = 10e-9
l = 5e-9
N = 100

Ec = El = E = 1 * eV
n = nn = np.arange(1, N + 1, 1)
kn = np.sqrt(El * 2 * m / (hbar ** 2) - (n * np.pi / (2 * w)) ** 2 + 0j)
knn = np.sqrt(Ec * 2 * m / (hbar ** 2) - (nn * np.pi / (2 * d)) ** 2 + 0j)
dn = [1 / np.sqrt(k) if np.isreal(k) else np.sqrt(2) for k in kn]
dnn = [1 / np.sqrt(2 * l) if np.isreal(k) else np.sqrt(2 * k * np.exp(2 * k * l) / (np.exp(4 * k * l) - 1)) for k in
       knn]

D = np.zeros((N, N), complex)
np.fill_diagonal(D, dn)


def f(m, nn):
    def fun(y, m, nn):
        return np.sin(nn * np.pi / (2 * d) * (y + d)) * np.sin(m * np.pi / (2 * w) * (y + w))

    return quad(fun, -w, w, args=(m, nn))[0] * dnn[nn] * np.exp(-1j * knn[nn] * l)


def ff(m, nn):
    def fun(y, m, nn):
        return np.sin(nn * np.pi / (2 * d) * (y + d)) * np.sin(m * np.pi / (2 * w) * (y + w))

    return quad(fun, -w, w, args=(m, nn))[0] * dnn[nn] * np.exp(1j * knn[nn] * l)


def g(m, n):
    def fun(y, m, n):
        return np.sin(n * np.pi / (2 * w) * (y + w)) * np.sin(m * np.pi / (2 * d) * (y + d))

    return quad(fun, -w, w, args=(m, n))[0] * dn[n] * kn[n]


F = FF = G = GAMMA = GAMMAGAMMA = np.zeros((N, N), complex)
for i in range(0, N):
    for j in range(0, N):
        F[i, j] = f(i, j)
        FF[i, j] = ff(i, j)
        G[i, j] = g(i, j)

np.fill_diagonal(GAMMA, dnn * knn * np.exp(-1j*knn*l))
np.fill_diagonal(GAMMAGAMMA, dnn * knn * np.exp(1j*knn*l))
