import random
import numpy as np

try:
    t = int(input("Enter t: "))
    if t < 6:
        raise ValueError("t must be greater at least 6")
    
except ValueError as exception:
    print(exception)
    exit()

e = 10 ** (-t)

def g_1 (x, y, h):
    return (3*f_1(x,y) - 4*f_1(x-h,y) + f_1(x-2*h,y)) / (2*h)

def g_2 (x, y, h):
    return (3*f_1(x,y) - 4*f_1(x,y-h) + f_1(x,y-2*h)) / (2*h)

def f_1 (x, y):
    return x**2 + y**2 - 2*x -4*y -1

def f_1_gradient(x, y):
    return [2*x - 2, 2*y - 4]

def f_2 (x, y):
    return 3*(x**2) - 12*x + 2*(y**2) + 16*y - 10

def f_2_gradient(x, y):
    return [6*x - 12, 4*y + 16]

def f_3 (x, y):
    return x**2 - 4*x*y + 5*(y**2) - 4*y + 3

def f_3_gradient(x, y):
    return [2*x - 4*y, -4*x + 10*y - 4]

def f_4 (x, y):
    return (x**2)*y - 2*x*(y**2) + 3*x*y + 4

def f_4_gradient(x, y):
    return [2*x*y - 2*(y**2) + 3*y, x**2 - 4*x*y + 3*x]

def function_minimization(e, calcul_rata, calcul_gradient):
    iterations = 0
    while True:
        x = np.random.uniform(-100, 100)
        y = np.random.uniform(-100, 100)
        k = 0
        while True:
            iterations += 1
            if calcul_gradient == 'Analitic':
                gradient = f_1_gradient(x, y)
            else:
                gradient = [g_1(x,y,10**(-5)), g_2(x, y,10**(-5))]
            if calcul_rata == 'Constant':
                rata = 10**(-3)
            else:
                beta = 0.8
                rata = 1
                p = 1
                while f_1(x - gradient[0], y - gradient[1]) > f_1(x, y) - rata / 2 * np.linalg.norm(gradient)**2 and p < 8:
                    rata = beta * rata
                    p += 1
            x = x - rata*gradient[0]
            y = y - rata*gradient[1]
            k += 1
            if rata * np.linalg.norm(gradient) < e or k > 30_000 or rata * np.linalg.norm(gradient) > 10**10:
                break
        if rata * np.linalg.norm(gradient) <= e:
            print("\nNumarul de itaratii: ", iterations)
            return [x, y]
        if k > 30_000:
            e = e * 10

print("Calculand gradientul functiei F folosind formula analitica si alegand o rata de invatare constanta obtinem urmatoarea aproximare a solutiei: \n", function_minimization(e, 'Constant', 'Analitic'))
print("Calculand gradientul functiei F folosind formula aproximarii si alegand o rata de invatare constanta obtinem urmatoarea aproximare a solutiei: \n", function_minimization(e, 'Constant', 'Aproximare'))
print("Calculand gradientul functiei F folosind formula analitica si calculand o rata de invatare variabila obtinem urmatoarea aproximare a solutiei: \n", function_minimization(e, 'Variabila', 'Analitic'))
print("Calculand gradientul functiei F folosind formula aproximarii si calculand o rata de invatare variabila obtinem urmatoarea aproximare a solutiei: \n", function_minimization(e, 'Variabila', 'Aproximare'))