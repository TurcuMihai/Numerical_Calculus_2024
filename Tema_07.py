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

def find_R(coefficients):
    return (np.abs(coefficients[0]) + max(np.abs(coefficients))) / np.abs(coefficients[0])
    
def horner(coefficients, x):
    n = len(coefficients)
    result = coefficients[0]
    for i in range(1, n):
        result = result * x + coefficients[i]
    return result

def muller_method(coefficients, n, e):
    roots = []
    iterations = 1
    while iterations < n:
        x_0 = np.random.uniform(-R, R) 
        x_1 = np.random.uniform(-R, R)
        k = 3
        ok = 0
        while ok == 0:
            x_0 = np.random.uniform(-R, R)
            x_1 = np.random.uniform(-R, R)
            x_2 = np.random.uniform(-R, R)
            computed_values = 1
            while True:
                h_0 = x_1 - x_0
                h_1 = x_2 - x_1
                delta_0 = (horner(coefficients, x_1) - horner(coefficients, x_0)) / h_0
                delta_1 = (horner(coefficients, x_2) - horner(coefficients, x_1)) / h_1
                a = (delta_1 - delta_0) / (h_1 + h_0)
                b = a * h_1 + delta_1
                c = horner(coefficients, x_2)
                if (b**2 - 4*a*c) < 0:
                    #print("Am intrat aici 1")
                    computed_values = 0
                    break
                if np.abs(max(b + np.sqrt(b**2 - 4*a*c), b - np.sqrt(b**2 - 4*a*c))) < e:
                    #print("Am intrat aici 2")
                    computed_values = 0
                    break
                delta_x = 2*c / max(b + np.sqrt(b**2 - 4*a*c), b - np.sqrt(b**2 - 4*a*c))
                x_3 = x_2 - delta_x
                k += 1
                x_0 = x_1
                x_1 = x_2
                x_2 = x_3
                if np.abs(delta_x) < e or k > 100 or np.abs(delta_x) > 10**8:
                    break
            if computed_values == 1:
                ok = 1

        if np.abs(delta_x) < e:
            roots.append(x_3)
            iterations += 1  
    return roots          


# coefficients = np.random.rand(n) 
# coefficients = [1.0, -6.0, 11.0, -6.0]
coefficients = [1.0, 1.0, -2.0]
print(f"\n The polynom is: ")
for i in range(n-1):
    print(f"{coefficients[i]}*x^{n-i-1}", end = " + ")
print(f"{coefficients[n-1]}*x^{0}")
R = find_R(coefficients)
print(f"\nThe interval [-R,R] is: [{-R}, {R}]")

roots = muller_method(coefficients, n, e)
print(roots)
for root in roots:
    print(horner(coefficients, root))
