import numpy as np
from scipy.linalg import block_diag

MaxN = 100
w1 = 10e-9
w2 = 10e-9
L = 5e-9
eV = 1.6e-19
hbar = 1.05e-34
m0 = 0.067 * 9.1e-31

# gamma
E = 0.1 * eV
val1 = [x + 0j for x in np.arange(1, MaxN) ** 2 * np.pi ** 2 / w1 ** 2 - 2 * E * m0 / hbar ** 2 if x < 0]
val2 = [x + 0j for x in np.arange(1, MaxN) ** 2 * np.pi ** 2 / w2 ** 2 - 2 * E * m0 / hbar ** 2 if x < 0]
assert(len(val1) == len(val2))
N = len(val1)
gamma1 = gamma2 = np.array(np.sqrt(val1)).reshape(N, 1)
gamma3 = gamma4 = np.array(np.sqrt(val2)).reshape(N, 1)

M1 = np.zeros((N, N), complex)
np.fill_diagonal(M1, np.exp(gamma2 * w2 / 2))

M2 = np.zeros((N, N), complex)
np.fill_diagonal(M2, np.exp(-gamma2 * w2 / 2))

M3 = np.zeros((N, N), complex)
np.fill_diagonal(M3, np.sinh(gamma1 * w2))

M4 = np.zeros((N, N), complex)
np.fill_diagonal(M4, gamma2 * np.exp(gamma2 * w2 / 2))

M5 = np.zeros((N, N), complex)
np.fill_diagonal(M5, gamma2 * np.exp(-gamma2 * w2 / 2))

M6 = np.zeros((N, N), complex)
np.fill_diagonal(M6, gamma1 * np.cosh(gamma1 * w2))

M7 = np.zeros((N, N), complex)
for n in range(N):
    for m in range(N):
        M7[n, m] = 2 / w1 * (m + 1) * np.pi / w2 * np.cos((m + 1) * np.pi) * w1 ** 2 * gamma3[m] ** 2 / (
                w1 ** 2 * gamma3[m] ** 2 + (n + 1) ** 2 * np.pi ** 2) * np.cos((n + 1) * np.pi) * np.sinh(
            gamma3[m] * w1) * (n + 1) * np.pi / (gamma3[m] ** 2 * w1)

M8 = np.zeros((N, N), complex)
for n in range(N):
    for m in range(N):
        M8[n, m] = 2 / w1 * (m + 1) * np.pi / w2 * np.cos((m + 1) * np.pi) * w1 ** 2 * gamma4[m] ** 2 / (
                w1 ** 2 * gamma4[m] ** 2 + (n + 1) ** 2 * np.pi ** 2) * np.cos((n + 1) * np.pi) * np.sinh(
            gamma4[m] * w1) * (n + 1) * np.pi / (gamma4[m] ** 2 * w1)

M9 = np.zeros((N, N), complex)
np.fill_diagonal(M9, np.exp(-gamma1 * w2 / 2))

M10 = np.zeros((N, N), complex)
np.fill_diagonal(M10, np.exp(gamma1 * w2 / 2))

M11 = np.zeros((N, N), complex)
np.fill_diagonal(M11, np.sinh(-gamma2 * w2))

M12 = np.zeros((N, N), complex)
np.fill_diagonal(M12, gamma1 * np.exp(-gamma1 * w2 / 2))

M13 = np.zeros((N, N), complex)
np.fill_diagonal(M13, gamma1 * np.exp(-gamma1 * w2 / 2))

M14 = np.zeros((N, N), complex)
np.fill_diagonal(M14, gamma2 * np.cosh(-gamma2 * w2))

M15 = np.zeros((N, N), complex)
for n in range(N):
    for m in range(N):
        M15[n, m] = 2 / w1 * (m + 1) * np.pi / w2 * w1 ** 2 * gamma3[m] ** 2 / (
                w1 ** 2 * gamma3[m] ** 2 + (n + 1) ** 2 * np.pi ** 2) * np.cos((n + 1) * np.pi) * np.sinh(
            gamma3[m] * w1) * (n + 1) * np.pi / (gamma3[m] ** 2 * w1)

M16 = np.zeros((N, N), complex)
for n in range(N):
    for m in range(N):
        M16[n, m] = 2 / w1 * (m + 1) * np.pi / w2 * w1 ** 2 * gamma4[m] ** 2 / (
                w1 ** 2 * gamma4[m] ** 2 + (n + 1) ** 2 * np.pi ** 2) * np.cos((n + 1) * np.pi) * np.sinh(
            gamma4[m] * w1) * (n + 1) * np.pi / (gamma4[m] ** 2 * w1)

M17 = np.zeros((N, N), complex)
np.fill_diagonal(M17, np.sinh(-gamma3 * L))

M18 = np.zeros((N, N), complex)
np.fill_diagonal(M18, np.sinh(gamma4 * w1))

M19 = np.zeros((N, N), complex)
np.fill_diagonal(M19, gamma3 * np.cosh(-gamma3 * L))

M20 = np.zeros((N, N), complex)
np.fill_diagonal(M20, gamma4 * np.cosh(gamma4 * w1))

M21 = np.zeros((N, N), complex)
for n in range(N):
    for m in range(N):
        M21[n, m] = 2 / w2 * (m + 1) * np.pi / w1 * np.cos((m + 1) * np.pi) * w2 ** 2 * gamma1[m] ** 2 / (
                w2 ** 2 * gamma1[m] ** 2 + (n + 1) ** 2 * np.pi ** 2) * np.cos((n + 1) * np.pi) * np.sinh(
            gamma1[m] * w2) * (n + 1) * np.pi / (gamma1[m] ** 2 * w2)

M22 = np.zeros((N, N), complex)
for n in range(N):
    for m in range(N):
        M22[n, m] = 2 / w2 * (m + 1) * np.pi / w1 * np.cos((m + 1) * np.pi) * w2 ** 2 * gamma2[m] ** 2 / (
                w2 ** 2 * gamma2[m] ** 2 + (n + 1) ** 2 * np.pi ** 2) * np.cos((n + 1) * np.pi) * np.sinh(
            gamma2[m] * w2) * (n + 1) * np.pi / (gamma2[m] ** 2 * w2)

M23 = np.zeros((N, N), complex)
np.fill_diagonal(M23, np.sinh(gamma4 * L))

M24 = np.zeros((N, N), complex)
np.fill_diagonal(M24, np.sinh(-gamma3 * w1))

M25 = np.zeros((N, N), complex)
np.fill_diagonal(M25, gamma4 * np.cosh(gamma4 * L))

M26 = np.zeros((N, N), complex)
np.fill_diagonal(M26, gamma3 * np.cosh(-gamma3 * w1))

M27 = np.zeros((N, N), complex)
for n in range(N):
    for m in range(N):
        M27[n, m] = 2 / w2 * (m + 1) * np.pi / w1 * w2 ** 2 * gamma1[m] ** 2 / (
                w2 ** 2 * gamma1[m] ** 2 + (n + 1) ** 2 * np.pi ** 2) * np.cos((n + 1) * np.pi) * np.sinh(
            gamma1[m] * w2) * (n + 1) * np.pi / (gamma1[m] ** 2 * w2)

M28 = np.zeros((N, N), complex)
for n in range(N):
    for m in range(N):
        M28[n, m] = 2 / w2 * (m + 1) * np.pi / w1 * w2 ** 2 * gamma2[m] ** 2 / (
                w2 ** 2 * gamma2[m] ** 2 + (n + 1) ** 2 * np.pi ** 2) * np.cos((n + 1) * np.pi) * np.sinh(
            gamma2[m] * w2) * (n + 1) * np.pi / (gamma2[m] ** 2 * w2)

assert (M1.shape == M2.shape == M3.shape == M4.shape == M5.shape == M6.shape == M7.shape == M8.shape == M9.shape ==
        M10.shape == M11.shape == M12.shape == M13.shape == M14.shape == M15.shape == M16.shape == M17.shape ==
        M18.shape == M19.shape == M20.shape == M21.shape == M22.shape == M23.shape == M24.shape == M25.shape ==
        M26.shape == M27.shape == M28.shape)
M0 = np.zeros(M1.shape)
row1 = np.concatenate((M3, M0, M0, M0, -1*M1, M0, M0, M0), axis=1)
row2 = np.concatenate((M6, M0, M7, M8, -1*M4, M0, M0, M0), axis=1)
row3 = np.concatenate((M0, M11, M0, M0, M0, -1*M10, M0, M0), axis=1)
row4 = np.concatenate((M0, M14, M15, M16, M0, M13, M0, M0), axis=1)
row5 = np.concatenate((M0, M0, M0, M18, M0, M0, -1*M17, M0), axis=1)
row6 = np.concatenate((M21, M22, M0, M20, M0, M0, -1*M19, M0), axis=1)
row7 = np.concatenate((M0, M0, M24, M0, M0, M0, M0, -1*M23), axis=1)
row8 = np.concatenate((M27, M28, M26, M0, M0, M0, M0, -1*M25), axis=1)
A = np.concatenate((row1, row2, row3, row4, row5, row6, row7, row8))

B = block_diag(M2, -1*M5, M9, M12, M0, M0, M0, M0)

assert(A.shape == B.shape)
b2 = np.zeros(gamma2.shape)
a1 = np.ones(gamma1.shape)
c0 = np.zeros(gamma3.shape)
y = np.concatenate((b2, b2, a1, a1, c0, c0, c0, c0))

rhs = B@y
x = np.linalg.solve(A, rhs)
da = x[0:N]
db = x[N:2*N]
dc = x[2*N:3*N]
dd = x[3*N:4*N]
a2 = x[4*N:5*N]
b1 = x[5*N:6*N]
c3 = x[6*N:7*N]
c4 = x[7*N:8*N]
T = np.sum(a2*np.conjugate(a2))/np.sum(a1*np.conjugate(a1))
print(T)
