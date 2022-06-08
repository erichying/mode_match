import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
eV = 1.6e-19
hbar = 1.05e-34
m0 = 0.067 * 9.1e-31
w1 = 50e-9
w2 = 20e-9
N = 100


def calculate_transmission(n):
    E = n * eV
    val1 = [x + 0j for x in np.arange(1, N) ** 2 * np.pi ** 2 / w1 ** 2 - 2 * E * m0 / hbar ** 2 if x < 0]
    val2 = [x + 0j for x in np.arange(1, N) ** 2 * np.pi ** 2 / w2 ** 2 - 2 * E * m0 / hbar ** 2 if x < 0]
    k1 = np.array(np.sqrt(val1))
    k2 = np.array(np.sqrt(val2))

    def integrand1(y, n, k):
        return np.sin(n * np.pi / w1 * (y - w1 / 2)) * np.sin(k * np.pi / w2 * (y - w2 / 2))

    def integrand2(y, m, k):
        return np.sin(m * np.pi / w2 * (y - w2 / 2)) * np.sin(k * np.pi / w1 * (y - w1 / 2))

    i1 = np.zeros((len(k2), len(k1)), complex)
    for n in range(1, len(k1) + 1):
        for k in range(1, len(k2) + 1):
            i1[k - 1, n - 1] = 2 / w2 * quad(integrand1, -w1 / 2, w1 / 2, args=(n, k))[0]

    i2 = np.zeros((len(k1), len(k2)), complex)
    for m in range(1, len(k2) + 1):
        for k in range(1, len(k1) + 1):
            i2[k - 1, m - 1] = 2 / w1 * quad(integrand2, -w1 / 2, w1 / 2, args=(m, k))[0] * k2[m - 1]

    kk = np.diag(k1)

    b_row1 = np.concatenate((i1, np.zeros((len(k2), len(k1)))), axis=1)
    b_row2 = np.concatenate((np.zeros((len(k1), len(k1))), kk), axis=1)
    B = np.concatenate((b_row1, b_row2))

    a = np.zeros((len(k1), 1))
    a[0] = 1
    rhs = np.concatenate((a, a))
    b = B@rhs

    a_row1 = np.concatenate((np.eye(len(k2)), -1*i1), axis=1)
    a_row2 = np.concatenate((i2, kk), axis=1)
    aa = np.concatenate((a_row1, a_row2))

    x = np.linalg.solve(aa, b)
    c = x[0:len(k2)]
    b = x[len(k2):len(k1)+len(k2)]

    return np.sum(np.abs(c))


val = []
for i in np.arange(0.1, 0.7, 0.001):
    val.append(calculate_transmission(i))

plt.plot(val)
plt.show()
