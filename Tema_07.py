import numpy as np
import random
import sys
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
    try:
        for i in range(1, n):
            result = result * x + coefficients[i]
        return result
    except OverflowError:
        print(f"Error: The result of the polynomial for x = {x} is too large to be represented.")
        return float('inf') 


def muller_method(coefficients, n, e, R):
    roots = []
    i = 0
    left_bound = -R
    increment = 0.05
    while left_bound <= R:
        x_0 = random.uniform(left_bound, left_bound + increment) 
        x_1 = random.uniform(left_bound, left_bound + increment)
        x_2 = random.uniform(left_bound, left_bound + increment)
        delta_x = 0
        computed_values = 0
        while True:
                k = 3
                h_0 = x_1 - x_0
                h_1 = x_2 - x_1
                if (np.abs(h_0) < e or np.abs(h_1) < e or np.abs(h_1 + h_0) < e):
                    break
                if (horner(coefficients, x_1) == float('inf') or horner(coefficients, x_0)) == float('inf') or horner(coefficients, x_2) == float('inf'):
                    break
                delta_0 = (horner(coefficients, x_1) - horner(coefficients, x_0)) / h_0
                delta_1 = (horner(coefficients, x_2) - horner(coefficients, x_1)) / h_1
                a = (delta_1 - delta_0) / (h_1 + h_0)
                b = a * h_1 + delta_1
                c = horner(coefficients, x_2)
                if b <= 0:
                    sign_b = -1
                else:
                    sign_b = 1
                if b >= np.sqrt(sys.float_info.max)/10**5 or a >= np.sqrt(sys.float_info.max)/10**5 or c >= np.sqrt(sys.float_info.max)/10**5:
                    break
                if b**2 <= 4*a*c:
                    break
                if np.abs(b + sign_b*np.sqrt(b**2 - 4*a*c)) < e:
                    break
                delta_x = 2*c / (b + sign_b*np.sqrt(b**2 - 4*a*c))
                x_3 = x_2 - delta_x
                k += 1
                x_0 = x_1
                x_1 = x_2
                x_2 = x_3
                computed_values = 1
                if np.abs(delta_x) < e or k > 1000 or np.abs(delta_x) > 10**8:
                    break
        if np.abs(delta_x) < e and computed_values == 1:
            found = 0
            for root in roots:
                if np.abs(x_3 - root) < e:
                    found = 1
                    break
            if found == 0:
                roots.append(x_3)
        left_bound += increment  
    return roots          


coefficients = [np.random.uniform(-10,10) for i in range(n)]
# coefficients = [1.0, -6.0, 11.0, -6.0]
# coefficients = [1.0, -55.0/42.0, -1.0, 49.0/42.0, -6.0/42.0]
# coefficients = [1.0, -38.0/8.0, 49.0/8.0, -22.0/8.0, 3.0/8.0]
# coefficients = [1.0, -6.0, 13.0, -12.0, 4.0]
print(f"\n The polynom is: ")
for i in range(n-1):
    print(f"{coefficients[i]}*x^{n-i-1}", end = " + ")
print(f"{coefficients[n-1]}*x^{0}")
R = find_R(coefficients)
print(f"\nThe interval [-R,R] is: [{-R}, {R}]")

roots = muller_method(coefficients, n, e, R)

with open("roots.txt", "a") as file:
    file.write(f"{coefficients}: ")

ok = 0
for root in roots:
    if np.abs(horner(coefficients, root)) < e:
        ok = 1
        print("O radacina reala a polinomului este: ", root, " , iar valoarea polinomului in aceasta radacina este: ", horner(coefficients, root))
        with open("roots.txt", "a") as file:
            file.write(str(root) + " ")
if ok == 0:
    with open("roots.txt", "a") as file:
        file.write(" The polynom has not real roots.\n")
    print(" The polynom has not real roots.")

#-------------------BONUS-------------------
#x* = -3.18306301193336   
def f_x(x):
    if x <= 250:
        return np.exp(x) - np.sin(x)
    else:
        print(f"Error: For x = {x}, function return a number too large to be represented.")
        exit()

def f_prime_x(x):
    if x <= 250:
        return np.exp(x) - np.cos(x)
    else:
        print(f"Error: For x = {x}, function return a number too large to be represented.")
        exit()

def newton(e):
    while True:
        x_0 = np.random.uniform(-3,3)
        # x_0 = -3
        gasit = 0
        while True:
            if (np.abs(f_prime_x(x_0)) < e):
                print("The derivative is 0")
                break
            y_0 = x_0 - f_x(x_0)/f_prime_x(x_0)
            if (np.abs(f_x(x_0)-f_x(y_0)) < e):
                break
            x_1 = x_0 - (f_x(x_0)**2 + f_x(y_0)**2) / f_prime_x(x_0)*(f_x(x_0)-f_x(y_0))
            if np.abs(x_1 - x_0) < e:
                gasit = 1
                break
            x_0 = x_1
        if gasit == 1:
            return x_1

x_star = newton(e)
print("\nSolutia aproximata de formula N4 este: ", x_star)
print("Valoarea functiei, folosind aceasta solutie este: ", f_x(x_star))


