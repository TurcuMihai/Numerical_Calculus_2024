import numpy as np
from numpy.linalg import det

try:
    n = int(input("Enter n: "))
    t = int(input("Enter t: "))
    if t < 5:
        raise ValueError("t must be greater at least 5")
    if n < 1:
        raise ValueError("n must be greater than 0")
except ValueError as exception:
    print(exception)
    exit()

e = 10 ** (-t)

def indexare(i, j):
    if i >= j:
        return i * (i + 1) // 2 + j
    else:
        return j * (j + 1) // 2 + i

def crout(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)
    try:
        for j in range(n):
            for i in range(j, n):
                L[i, j] = A[i, j] - np.sum(L[i, :j] * U[:j, j])
            for i in range(j+1, n):
                if np.abs(L[j, j]) > e: 
                    U[j, i] = (A[j, i] - np.sum(L[j, :i] * U[:i, i])) / L[j, j]
                else:
                    raise Exception(f"LU descomposition is not possible, because det(A_{j+1}) = 0")
    except Exception as exception:
        print(exception)
        exit()
    return L, U


def crout_restricted(A):
    n = len(A)
    try:
        for p in range(n):
            for i in range(p, n):
                    A[i, p] = A_init[i, p] - np.sum(A[i, :p] * A[:p, p])
            for i in range(p+1, n):
                if np.abs(A[p, p]) > e:
                        A[p, i] = (A_init[p,i] - np.sum(A[p, :p] * A[:p, i])) / A[p, p]
                else:
                    raise Exception(f"LU descomposition is not possible, because det(A_{p+1}) = 0")
    except Exception as exception:
        print(exception)
        exit()
    return A
    

print(f"Parametrii programului sunt: n = {n}, t = {t}, precession e={e}\n")
 
# A = np.random.rand(n, n)
# b = np.random.rand(n)
A = np.array([[2.5, 2, 2], [5, 6, 5], [5, 6, 6.5]])
b = np.array([2,2,2])
# A = np.array([[2, 3, 2], [2, 3, 5], [5, 6, 6.5]])
# b = np.array([2,2,2])

A_init = A.copy()
b_init = b.copy()
 
L, U = crout(A)
print("LU decomposition using Crout's method: ")
print("A = \n", A)
print("L = \n", L)
print("U = \n", U)

A = crout_restricted(A)
print("\ndecomposition of A = \n", A)

print("\ndet(A) = det(A_init)")
det_A = 1
for i in range(len(A)):
    det_A = det_A * A[i, i]
print(A_init.shape)
print(det_A, "=", det(A_init))

#-----------------Crout method-----------------#
# # Forward substitution for Ly = b
# y = np.zeros_like(b, dtype=float)
# for i in range(n):
#     y[i] = b[i] - np.sum(L[i, :i] * y[:i])
#     if np.abs(L[i, i]) > e: 
#         y[i] /= L[i, i]
#     else:
#         print(f"Division by zero detected in forward substitution at i={i}, L[i, i]={L[i, i]}")
#         exit()
 
# # Backward substitution for Ux = y
# x_LU = np.zeros_like(y, dtype=float)
# for i in range(n-1, -1, -1):
#     x_LU[i] = y[i] - np.sum(U[i, i+1:] * x_LU[i+1:])
#     if np.abs(U[i, i]) > e:
#         x_LU[i] /= U[i, i]
#     else:
#         print(f"Division by zero detected in backward substitution at i={i}, U[i, i]={U[i, i]}")
#         exit()
#-----------------Crout restricted--------------#
# Forward substitution for Ly = b
y = np.zeros_like(b, dtype=float)
for i in range(n):
    y[i] = b[i] - np.sum(A[i, :i] * y[:i])
    if np.abs(A[i, i]) > e: 
        y[i] /= A[i, i]
    else:
        print(f"Division by zero detected in forward substitution at i={i}, L[i, i]={L[i, i]}")
        exit()

# Backward substitution for Ux = y
x_LU = np.zeros_like(y, dtype=float)
for i in range(n-1, -1, -1):
    x_LU[i] = y[i] - np.sum(A[i, i+1:] * x_LU[i+1:])
    if np.abs(A[i, i]) > e:
        x_LU[i] /= 1
    else:
        print(f"Division by zero detected in backward substitution at i={i}, U[i, i]={U[i, i]}")
        exit()

print("\nThe solution of the system is: ")
print(x_LU)


print("\nVerify the solution: ")
norm = np.linalg.norm(np.dot(A_init, x_LU) - b_init)
print(f"||A*x_LU - b||2 = {norm}")
print(f"norm < 1e-9:", norm < 10 ** (-9))

print("\nThe norms are:")

x_lib = np.linalg.solve(A, b)
A_inv = np.linalg.inv(A_init)

norm1 = np.linalg.norm(x_LU - x_lib)
norm2 = np.linalg.norm(x_LU - np.dot(A_inv, b_init))

print(f"||x_LU - x_lib||2 = {norm1}")
print(f"||x_LU - A^(-1)*b_init||2 = {norm2}")

#----------------------BONUS----------------------#

def LU_optimized(A):
    n = len(A)
    L = np.ones(n*(n+1)//2)
    U = np.ones(n*(n+1)//2)
    try:
        for j in range(n):
            for i in range(j, n):
                L[indexare(i,j)] = A[i, j] - sum(L[indexare(i,k)] * U[indexare(k,j)] for k in range(j))
            for i in range(j+1, n):
                if np.abs(L[indexare(j,j)]) > e: 
                    U[indexare(j,i)] = (A[j, i] - sum((L[indexare(j,k)] * U[indexare(k,i)]) for k in range(j))) / L[indexare(j,j)]
                else:
                    raise Exception(f"LU descomposition is not possible, because det(A_{j+1}) = 0")
    except Exception as exception:
        print(exception)
        exit()
    return L, U

L, U = LU_optimized(A_init)

print("Vectorul L este:", L)
print("Vectorul U este:", U)

# Forward substitution for Ly = b
y = np.zeros_like(b, dtype=float)
for i in range(n):
    y[i] = b[i] - sum(L[indexare(i,k)] * y[k] for k in range(i))
    if np.abs(L[indexare(i,i)]) > e: 
        y[i] /= L[indexare(i,i)]
    else:
        print(f"Division by zero detected in forward substitution at i={i}, L[i, i]={L[i, i]}")
        exit()
 
# Backward substitution for Ux = y
x_LU = np.zeros_like(y, dtype=float)
for i in range(n-1, -1, -1):
    x_LU[i] = y[i] - sum(U[indexare(i,k)] * x_LU[k] for k in range(i+1, n))
    if np.abs(U[indexare(i,i)]) > e:
        x_LU[i] /= U[indexare(i,i)]
    else:
        print(f"Division by zero detected in backward substitution at i={i}, U[i, i]={U[i, i]}")
        exit()
print(x_LU)