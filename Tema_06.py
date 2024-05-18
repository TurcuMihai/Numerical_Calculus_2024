import matplotlib.pyplot as plt
import numpy as np

try:
    n = int(input("Enter n: "))
    x_0 = float(input("Enter x_0: "))
    x_n = float(input("Enter x_n: "))
    if x_0 >= x_n:
        raise ValueError("x_0 must be less than x_n")
    if n < 3:
        raise ValueError("n must be greater than 2")
except ValueError as exception:
    print(exception)
    exit()

def f_x(x):
    return x**4 - 12*(x**3) + 30*(x**2) + 12
    #return x**2 + 1

def compute_t (x_tilda, x, h):
    return (x_tilda - x) / h

def compute_factor(t, i, factor_anterior):
    return factor_anterior * (t - i + 1) / i

def compute_value_in_function(y, t, n):
    result = y[0]
    factor_anterior = 1
    for i in range(1, n+1):
        factor = compute_factor(t, i, factor_anterior)
        result += y[i] * factor
        factor_anterior = factor
    return result

def least_squares_method(x, y, n):
    print(x)
    print(y)
    m = 4  #gradul polinomului
    B = np.zeros((m+1, m+1))
    f = np.zeros(m+1)
    for i in range(m+1):
        for j in range(m+1):
            for k in range(n+1):
                B[i][j] += x[k]**(i+j)
        for k in range(n+1):
            f[i] += y[k] * (x[k]**i)
    return B, f

def horner(coefficients, x):
    n = len(coefficients)
    result = coefficients[n-1]
    for i in range(n-2, -1, -1):
        result = result * x + coefficients[i]
    return result

# print(horner([12, 0, 30, -12, 4], 4))
# print(f_x(4))

# x = [0, 1, 2, 3, 4, 5]
# y = [50, 47, -2, -121, -310, -545]

h = (x_n - x_0) / n
x = [x_0]
y = [f_x(x_0)]
for i in range(1, n):
    x.append(x_0 + i * h)
    y.append(f_x(x[i]))
x.append(x_n)
y.append(f_x(x[n]))
x_tilda = np.random.uniform(x_0, x_n)
while x_tilda in x:
    x_tilda = np.random.uniform(x_0, x_n)
print("\nx_tilda= ", x_tilda)
print("\nx= ", x)
print("\ny= ", y)

delta = y
for i in range(1, n+1):
    for j in range(n, i-1, -1):
        delta[j] = (delta[j] - delta[j-1])
print("\ndelta: ", delta)

t = compute_t(x_tilda, x[0], h)
print("\nt = ", t)

print(f_x(x_tilda))
print(compute_value_in_function(y, t, n))
print(f"\nL_n({x_tilda}) = {compute_value_in_function(y, t, n)}")
print(f"\nf_x({x_tilda}) = {f_x(x_tilda)}")
print(f"\n| L_n({x_tilda}) - f({x_tilda}) | = {np.abs(compute_value_in_function(y, t, n) - f_x(x_tilda))}")


h = (x_n - x_0) / n
x = [x_0]
y = [f_x(x_0)]
for i in range(1, n):
    x.append(x_0 + i * h)
    y.append(f_x(x[i]))
x.append(x_n)
y.append(f_x(x[n]))

B, f = least_squares_method(x, y, n)
# print(np.dot(B,a))
print("\nB= ", B)
print("\nf= ", f)
if np.linalg.det(B) == 0:
    print("The matrix B is singular, cannot solve the system")
else:
    a = np.linalg.solve(B, f)
print(np.dot(B,a))
print("\na= ", a)

print(f"\nf_x(x_tilda) = ", f_x(x_tilda))
print(f"\nP_m(x_tilda) = ", horner(a, x_tilda))
print(f"\n| P_m(x_tilda) - f(x_tilda) | = ", np.abs(horner(a, x_tilda) - f_x(x_tilda)))
print(f"\n SUM: ", np.sum([abs(horner(a, x[i]) - y[i]) for i in range(n+1)]))


def compute_Pm(x, y, x_tilda, m):
    B, f = least_squares_method(x, y, m)
    a = np.linalg.solve(B, f)
    return horner(a, x_tilda)


h = (x_n - x_0) / n
x = [x_0]
y = [f_x(x_0)]
for i in range(1, n):
    x.append(x_0 + i * h)
    y.append(f_x(x[i]))
x.append(x_n)
y.append(f_x(x[n]))

delta = y
for i in range(1, n+1):
    for j in range(n, i-1, -1):
        delta[j] = (delta[j] - delta[j-1])
print("\ndelta: ", delta)


Ln_values = [compute_value_in_function(y, compute_t(x_tilda , x[0], h), n) for x_tilda in x]

h = (x_n - x_0) / n
x = [x_0]
y = [f_x(x_0)]
for i in range(1, n):
    x.append(x_0 + i * h)
    y.append(f_x(x[i]))
x.append(x_n)
y.append(f_x(x[n]))



Pm_values = [compute_Pm(x, y, x_tilda, n) for x_tilda in x]

# Trasam graficul
plt.figure(figsize=(10, 6))
plt.plot(x, y , label='f(x)', color='blue')
plt.plot(x, Ln_values, label=f'L_{n}(x)', linestyle='--', color='green')
plt.plot(x, Pm_values, label=f'P_{n}(x)', linestyle='-.', color='red')
plt.scatter(x, y, color='black', label='Data Points')
plt.scatter([x_tilda], [f_x(x_tilda)], color='red', label=f'$x_tilda$: {round(f_x(x_tilda), 2)}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Approximations of f(x)')
plt.legend()
plt.grid(True)
plt.show()