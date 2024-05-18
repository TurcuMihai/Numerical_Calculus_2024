import numpy as np

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
        for j in range(i, n):
            A_t[i, j] = A[j, i]
            A_t[j, i] = A[i, j]
    return A_t

def compute_b(A, s):
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += A[i, j] * s[j]
    return b
            
def householder(A, b, e):
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


def solve (R, b, e):
    sol = np.zeros_like(b, dtype=float)
    for i in range(n-1, -1, -1):
        sol[i] = b[i] - np.sum(R[i, i+1:] * sol[i+1:])
        if np.abs(R[i, i]) > e:
            sol[i] /= R[i, i]
        else:
            print(f"Division by zero detected in backward substitution at i={i}, R[i, i]={R[i, i]}")
            exit()
    return sol

def compute_inverse(R, Q):
    A_inv = np.zeros((n, n))
    for i in range(n):
        y = solve(R, Q[i], e)
        A_inv[i] = y
    return transpose(A_inv)

A = np.random.uniform(-100, 100, size=(n, n))
s = np.random.uniform(-100, 100, size=(n))
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

# compute the solution using the householder method
x_householder = solve(R, np.dot(Q.T, b_init), e)
print("\nx_householder : ", x_householder)

# compute the solution using the library function
Q_lib = np.linalg.qr(A_init)[0]
R_lib = np.linalg.qr(A_init)[1]
x_QR = np.linalg.solve(R_lib, np.dot(Q_lib.T, b_init))

print("\nx_QR : ", x_QR)
print("\n|| x_householder - x_QR || = ", np.linalg.norm(x_householder - x_QR))

print("\n|| A_init*x_householder - b_init || = ", np.linalg.norm(np.dot(A_init, x_householder) - b_init))
print("\n|| A_init*x_QR - b_init || = ", np.linalg.norm(np.dot(A_init, x_QR) - b_init))
print("\n|| x_householder - s || / ||s|| = ", np.linalg.norm(x_householder - s) / np.linalg.norm(s))
print("\n|| x_QR - s || / ||s|| = ", np.linalg.norm(x_QR - s) / np.linalg.norm(s))

print("\nA_inv_householder = \n", compute_inverse(R, Q))
print("\nA_inv_lib = \n", np.linalg.inv(A_init))
print("\n|| A_inv_householder - A_inv_lib || = ", np.linalg.norm(compute_inverse(R, Q) - np.linalg.inv(A_init)))

#----------------------BONUS----------------------
def multiply_upper_triangular(A, B):
    n = A.shape[0]
    result = np.zeros_like(B)
    for i in range(n):
        for j in range(n):
            # if i <= j:  # Verificăm dacă suntem în partea superioară triunghiulară a lui A
                result[i, j] = np.sum(A[i, i:] * B[i:, j])
    return result

def create_symetric_matrix(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A[i, j] = np.random.uniform(-100, 100)
            A[j, i] = A[i, j]
    return A

def compute_limit(A, b, e):
    k = 0
    A_ant = A.copy()
    b_init = b.copy()
    Q, R = householder(A, b, e)
    print()
    A = multiply_upper_triangular(R, Q)
    k += 1
    while np.linalg.norm(A - A_ant) > e:
        A_ant = A.copy()
        b = b_init.copy()
        Q, R = householder(A, b, e)
        A = multiply_upper_triangular(R, Q)
        k += 1
        if np.allclose(A, A_ant, atol=e):
            break
    return A

A = create_symetric_matrix(n)
s = np.random.uniform(-100, 100, size=(n))
print("\nA = \n", A)
b = compute_b(A, s)
print("\nThe limit of the sequence \n", compute_limit(A, b, e))
print("\n")


