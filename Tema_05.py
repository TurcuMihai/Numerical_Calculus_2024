import numpy as np

try:
    p = int(input("Enter p: "))
    n = int(input("Enter n: "))
    t = int(input("Enter t: "))
    if t < 6:
        raise ValueError("t must be greater at least 6")
    if n < 1:
        raise ValueError("n must be greater than 0")
    if p < 1:
        raise ValueError("p must be greater than 0")
except ValueError as exception:
    print(exception)
    exit()

e = 10 ** (-t)

def generate_positive_definite_matrix(n):
    while True:

        A = np.random.rand(n, n)
        

        A = 0.5 * (A + A.T)
        
        eps = 1e-10
        A += np.eye(n) * eps
        

        try:
            np.linalg.cholesky(A)
            return A 
        except np.linalg.LinAlgError:

            continue

def diagonal_matrix(A):
    n = len(A)
    for i in range(n):
        for j in range(n):
            if i != j and abs(A[i, j]) > e:
                return False
    return True


def Jacobi(A, n, e):
    k = 0
    k_max = 1000
    U = np.eye(n)
    p = 0
    q = 1
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i, j]) > abs(A[p, q]):
                p = i
                q = j
    alpha = (A[p, p] - A[q, q]) / (2*A[p, q])
    if np.abs(alpha) < e:
        t = -alpha - np.sqrt(alpha**2 + 1)
    else:
        t = -alpha + np.sqrt(alpha**2 + 1)
    c = 1 / np.sqrt(1 + t**2)
    s = t / np.sqrt(1 + t**2)

    while diagonal_matrix(A) == False and k <= k_max:
        for j in range(n):
            if j != p and j != q:
                A[p, j] = c * A[p, j] + s * A[q, j]
                A[q, j] = A[j, q] = -s * A[j, p] + c * A[q, j]
                A[j, p] = A[p, j]
            A[p, p] = A[p, p] + t*A[p, q]
            A[q, q] = A[q, q] - t*A[p, q]
            A[p, q] = A[q, p] = 0

        for i in range(n):
            save = U[i, p]
            U[i, p] = c * U[i, p] + s * U[i, q]
            U[i, q] = -s * save + c * U[i, q]
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > abs(A[p, q]):
                    p = i
                    q = j
        alpha = (A[p, p] - A[q, q]) / (2*A[p, q])
        if np.abs(alpha) < e:
            t = -alpha - np.sqrt(alpha**2 + 1)
        else:
            t = -alpha + np.sqrt(alpha**2 + 1)
        c = 1 / np.sqrt(1 + t**2)
        s = t / np.sqrt(1 + t**2)  
        k += 1 

    if k > k_max:
        print("The algorithm did not converge")
        exit()
    else:
        return A, U    


A = generate_positive_definite_matrix(n)
# A = np.array([[4, -1, 2], [-1, 5, 3], [2, 3, 6]])
print("\nA= ")
print(A)
A_bonus = np.copy(A)
A_init = np.copy(A)
A, U = Jacobi(A, n, e)
print("\nA_jacobi= ")
print(A)
print("\nU= ")  
print(U)  

lam = []
for i in range(n):
    lam.append(A[i, i])
print("\nEigenvalues: ")
print(lam)

print("\n|| A_init*U - U*lambda|| = ", np.linalg.norm(np.dot(A_init, U) - np.dot(U, np.diag(lam))))

#-----------Ex 2----------------

def compute_sequence(A_0, L_0, n, e): 
    k = 0
    k_max = 1000
    while True:
        A_1 = L_0.T * L_0
        L_1 = np.linalg.cholesky(A_1)
        k+=1
        if np.linalg.norm(A_1 - A_0) < e or k > k_max:
            break
        A_0 = A_1
        L_0 = L_1
    if k > k_max:
        print("The algorithm did not converge")
        exit()
    else:
        return A_1, L_1

print(A_init)
A = A_init
L = np.linalg.cholesky(A)
A_k, L_k = compute_sequence(A, L, n, e)
print("\nA_k= ")
print(A_k)


#-----------Ex 3----------------

def compute_matrix(n, p):
    A = np.zeros((p, n))
    for i in range(p):
        for j in range(n):
            A[i, j] = np.random.uniform(-100, 100)
    return A


A = compute_matrix(n, p)
# A= [[1,1],[1,0], [0,1]]
# A=[[1,2,3],[6,4,5],[-2,4,1],[32,4,21]]
A_init = np.copy(A)
print("\nA= ")
print(A)


U, S, V = np.linalg.svd(A)
print("\nU=  ", U)
print("\nS=  ", S)
print("\nV=  ", V)
singular_values = [S[i] for i in range(len(S)) if abs(S[i]) >= e]
print("\nSingular values= ")
print(singular_values)
print("\nS= ", S)

rang_A = 0
for i in range(len(S)):
    if abs(S[i]) > e:
        rang_A += 1
print("\nRank of A= ", rang_A)

print("\nRank of A using library= ", np.linalg.matrix_rank(A))

sigma_max = max(singular_values)

sigma_min = max(singular_values)
for i in range(len(S)):
    if abs(S[i]) > e and sigma_min > S[i]:
        sigma_min = S[i]
print("\nCondition number= ", sigma_max/sigma_min)

print("\nCondition number using library= ", np.linalg.cond(A))

S_i = np.zeros((n, p))
singular_values = [S[i] for i in range(len(S)) if abs(S[i]) > e]
for i in range(len(singular_values)):
    S_i[i, i] = 1 / singular_values[i]
print("\nS_i= ")
print(S_i)
A_i = np.dot(np.dot(V.T, S_i), U.T)
print("\nA_i= ")
print(A_i)

print("\nA_init= ")
print(A_init)
A_j = np.dot(np.linalg.inv(np.dot(A_init.T, A_init)), A_init.T)
print("\nA_j= ")
print(A_j)

print("\n|| A_i - A_j|| = ", np.sum(np.abs(A_i - A_j)))
print("\n|| A_i - A_j|| = ", np.linalg.norm(A_i - A_j))


#-----------BONUS----------------

def restricted_diagonal_matrix(A):
    for i in range(n):
        for j in range(n):
            if i != j and abs(A[indexare(i, j)]) > e:
                return False
    return True

def indexare(i, j):
    if i >= j:
        return i * (i + 1) // 2 + j
    else:
        return j * (j + 1) // 2 + i

def Jacobi_restricted(A, n, e):

    k = 0
    k_max = 1000
    U = np.eye(n)
    p = 0
    q = 1
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[indexare(i, j)]) > abs(A[indexare(p, q)]):
                p = i
                q = j
    alpha = (A[indexare(p, p)] - A[indexare(q, q)]) / (2*A[indexare(p, q)])
    t = -alpha + (alpha**2 + 1)**0.5
    c = 1 / np.sqrt(1 + t**2)
    s = t / np.sqrt(1 + t**2)

    while restricted_diagonal_matrix(A) == False and k <=  k_max:
        for j in range(n):
            if j != p and j != q:
                ant = A[indexare(p, j)]
                A[indexare(p, j)] = c * A[indexare(p, j)] + s * A[indexare(q, j)]
                A[indexare(j, q)]  = -s * ant + c * A[indexare(q, j)]
            A[indexare(p, p)] = A[indexare(p, p)] + t*A[indexare(p, q)]
            A[indexare(q, q)] = A[indexare(q, q)] - t*A[indexare(p, q)]
            A[indexare(p, q)] = 0

        for i in range(n):
            save = U[i, p]
            U[i, p] = c * U[i, p] + s * U[i, q]
            U[i, q] = -s * save + c * U[i, q]
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[indexare(i, j)]) > abs(A[indexare(p, q)]):
                    p = i
                    q = j
        alpha = (A[indexare(p, p)] - A[indexare(q, q)]) / (2*A[indexare(p, q)])
        t = -alpha + (alpha**2 + 1)**0.5
        c = 1 / np.sqrt(1 + t**2)
        s = t / np.sqrt(1 + t**2)
        k += 1
    if k > k_max:
        print("The algorithm did not converge")
        exit()
    else:
        return A, U

def multiply(A, U):
    n = len(U)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i, j] += A[indexare(i, k)] * U[k, j]
    return result


A = np.random.uniform(-100, 100, n*(n+1) // 2)
for i in range(n):
    for j in range(i, n):
        A[indexare(i, j)] = A_bonus[i, j]
# A = np.array([4, -1, 5, 2, 3, 6])

A_init = np.copy(A)
A, U = Jacobi_restricted(A, n, e)
print("\nA_jacobi= ")
print(A)
print("\nU= ")  
print(U)  

lam = []
for i in range(n):
    lam.append(A[indexare(i, i)])
print("\nEigenvalues: ")
print(lam)

print("\n|| A_init*U - U*lambda|| = ", np.linalg.norm(multiply(A_init, U) - np.dot(U, np.diag(lam))))

