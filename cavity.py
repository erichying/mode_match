import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

eV = 1.6e-19
hbar = 1.05e-34
m = 0.067 * 9.1e-31
w = 5e-9
d = 10e-9
l = 5e-9
N = 100


def calculate_transmission(v):
    Ec = Elr = v * eV
    n = nn = np.arange(1, N + 1, 1)
    kn = np.sqrt(Elr * 2 * m / (hbar ** 2) - (n * np.pi / (2 * w)) ** 2 + 0j)
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

    np.fill_diagonal(GAMMA, dnn * knn * np.exp(-1j * knn * l))
    np.fill_diagonal(GAMMAGAMMA, dnn * knn * np.exp(1j * knn * l))

    M0 = np.zeros((N, N))
    row1 = np.concatenate((F, FF, -np.sqrt(d * w) * D, M0), axis=1)
    row2 = np.concatenate((FF, F, M0, -np.sqrt(d * w) * D), axis=1)
    row3 = np.concatenate((GAMMA, -1 * GAMMAGAMMA, G, M0), axis=1)
    row4 = np.concatenate((GAMMAGAMMA, -1 * GAMMA, M0, -1 * G), axis=1)

    A = np.concatenate((row1, row2, row3, row4))

    B = block_diag(np.sqrt(d * w) * D, M0, G, M0)

    assert (A.shape == B.shape)

    a0 = np.zeros((N, 1))
    a0[0] = 1
    y = np.concatenate((a0, np.zeros((N, 1)), a0, np.zeros((N, 1))))
    rhs = B @ y
    x = np.linalg.solve(A, rhs)

    C = x[0:N]
    CC = x[N:2 * N]
    R = x[2 * N:3 * N]
    T = x[3 * N:4 * N]

    transmission = sum(np.abs(T))
    reflection = sum(np.abs(R))
    return transmission, reflection


v = np.arange(0.1, 0.2, 0.001)
t = []
r = []
for i in v:
    transmission, reflection = calculate_transmission(i)
    t.append(transmission)
    r.append(reflection)

plt.plot(v, t)
plt.show()
