import numpy as np
from numpy.linalg import det

try:
    n = int(input("Enter n: "))
    t = int(input("Enter t: "))
    if t < 6:
        raise ValueError("t must be greater at least 6")
    if n < 1:
        raise ValueError("n must be greater than 0")
except ValueError as exception:
    print(exception)
    exit()

e = 10 ** (-t)

def transpose(A):
    n = len(A)
    A_t = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            A_t[i, j] = A[j, i]
            A_t[j, i] = A[i, j]
    return A_t

def compute_b(A, s):
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += A[i, j] * s[j]
    return b
            
def householder (A, b, e):
    u = np.zeros(n)
    Q_t = np.eye(n)
    for r in range(n-1):
        sigma = np.sum(A[r:n, r] ** 2)
        if sigma <= e:
            print("The matrix A is singular")
            break
        k = np.sqrt(sigma)
        if A[r, r] > 0:
            k = -k
        beta = sigma - k * A[r, r]
        u[r] = A[r, r] - k
        u[r+1:n] = A[r+1:n, r]
        for j in range(r+1, n):
            gamma = np.sum(u[r:n] * A[r:n, j]) / beta
            for i in range(r, n):
                A[i, j] -= gamma * u[i]
        A[r, r] = k
        A[r+1:n, r] = 0
        gamma = np.sum(u[r:n] * b[r:n]) / beta
        for i in range(r, n):
            b[i] -= gamma * u[i]
        for j in range(n):
            gamma = np.sum(u[r:n] * Q_t[r:n, j]) / beta
            for i in range(r, n):
                Q_t[i, j] -= gamma * u[i]
    Q = transpose(Q_t)
    return Q, A

def solve (R, Q, b):
    Q_t = transpose(Q)
    result = np.dot(Q_t, b)
    n = len(R)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = result[i]
        for j in range(i+1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    return x

def compute_inverse(R, Q):
# estimate the inverse of A using R and Q decompositions
    A_inv = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        x = solve(R, Q, e_i)
        for j in range(n):
            A_inv[j, i] = x[j]
    return A_inv

# A = np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]])
# s = np.array([3, 2, 1])
A = np.random.rand(n, n)
s = np.random.rand(n)
print("\nA = ")
print(A)
b = compute_b(A, s)
print("\nb = ")
print(b)
A_init = A.copy()
b_init = b.copy()

Q, R = householder(A, b, e)
print("\nQ = ")
print(Q)
print("\nR = ")
print(R)

x_householder = solve(R, Q, b_init)
print(x_householder)

x_QR = np.linalg.solve(R, np.dot(transpose(Q), b_init))
print(x_QR)

print("\n|| x_QR - x_householder || = ", np.linalg.norm(x_QR - x_householder))

print("\n|| A_init*x_householder - b_init || = ", np.linalg.norm(np.dot(A_init, x_householder) - b_init))
print("\n|| A_init*x_QR - b_init || = ", np.linalg.norm(np.dot(A_init, x_QR) - b_init))
print("\n|| x_householder - s || / ||s|| = ", np.linalg.norm(x_householder - s) / np.linalg.norm(s))
print("\n|| x_QR - s || / ||s|| = ", np.linalg.norm(x_QR - s) / np.linalg.norm(s))

print("\nA_inv = \n", compute_inverse(R, Q))
print("\n|| A_inv_householder - A_inv_QR || = ", np.linalg.norm(compute_inverse(R, Q) - np.linalg.inv(A_init)))

