import math
import random
from collections import Counter

def find_the_minimum():
    m = 0
    u = 10**(-m)
    while (1 + u != 1):
        m = m + 1
        u = 10**(-m)
    return u*10

print("\nPrecizia masina este:", find_the_minimum(), "\n")


def neasociativity():
    x = 1.0
    y = find_the_minimum()/10
    z = find_the_minimum()/10
    print("Neasociativitatea adunarii:")
    print((x + y) + z, "=", x + (y + z))
    print((x + y) + z == x + (y + z))
    print("\n")
    x = find_the_minimum()
    y = find_the_minimum()
    z = 1_000_0000_0000_000_000_000.0
    print("Neasociativitatea inmultirii:")
    print(x * (y * z), "=", (x * y) * z)
    print((x * y) * z == x * (y * z))
neasociativity()


def tan_4(a):
    return (105*a - 10*(a**3)) / (105 - 45*(a**2) + a**4)

def tan_5(a):
    return (945*a - 105*(a**3) + a**5) / (945 - 420*(a**2) + 15*(a**4))

def tan_6(a):
    return (10395*a - 1260*(a**3) + 21*(a**5)) / (10395 - 4725*(a**2) + 210*(a**4) - a**6)

def tan_7(a):
    return (135135*a - 17325*(a**3) + 378*(a**5) - a**7) / (135135 - 62370*(a**2) + 3150*(a**4) - 28*(a**6))

def tan_8(a):
    return (2027025*a - 270270*(a**3) + 6930*(a**5) - 36*(a**7)) / (2027025 - 945945*(a**2) + 51975*(a**4) - 630*(a**6) + a**8)

def tan_9(a):
    return (34459425*a - 4729725*(a**3) + 135135*(a**5) - 990*(a**7) + a**9) / (34459425 - 16216200*(a**2) + 945945*(a**4) - 13860*(a**6) + 45*(a**8))

random_numbers = [random.uniform(-math.pi/2, math.pi/2) for _ in range(10000)]


tan_4_estimations = {a: tan_4(a) for a in random_numbers}
tan_5_estimations = {a: tan_5(a) for a in random_numbers}
tan_6_estimations = {a: tan_6(a) for a in random_numbers}
tan_7_estimations = {a: tan_7(a) for a in random_numbers}
tan_8_estimations = {a: tan_8(a) for a in random_numbers}
tan_9_estimations = {a: tan_9(a) for a in random_numbers}

exact_values = {a: math.tan(a) for a in random_numbers}

error_tan_4 = {a: abs(tan_4_estimations[a] - exact_values[a]) for a in random_numbers}
error_tan_5 = {a: abs(tan_5_estimations[a] - exact_values[a]) for a in random_numbers}
error_tan_6 = {a: abs(tan_6_estimations[a] - exact_values[a]) for a in random_numbers}
error_tan_7 = {a: abs(tan_7_estimations[a] - exact_values[a]) for a in random_numbers}
error_tan_8 = {a: abs(tan_8_estimations[a] - exact_values[a]) for a in random_numbers}
error_tan_9 = {a: abs(tan_9_estimations[a] - exact_values[a]) for a in random_numbers}

best_estimation_tan = {}

for a in random_numbers:
    best_estimation = error_tan_4[a]
    best_estimation_tan[a] = "tan_4"
    if error_tan_5[a] < best_estimation:
        best_estimation = error_tan_5[a]
        best_estimation_tan[a] = "tan_5"
    if error_tan_6[a] < best_estimation:
        best_estimation = error_tan_6[a]
        best_estimation_tan[a] = "tan_6"
    if error_tan_7[a] < best_estimation:
        best_estimation = error_tan_7[a]
        best_estimation_tan[a] = "tan_7"
    if error_tan_8[a] < best_estimation:
        best_estimation = error_tan_8[a]
        best_estimation_tan[a] = "tan_8"
    if error_tan_9[a] < best_estimation:
        best_estimation = error_tan_9[a]
        best_estimation_tan[a] = "tan_9"

def sin(n,a):
    if n == 4:
        return tan_4(a) / math.sqrt(1 + tan_4(a)**2)
    elif n == 5:
        return tan_5(a) / math.sqrt(1 + tan_5(a)**2)
    elif n == 6:
        return tan_6(a) / math.sqrt(1 + tan_6(a)**2)
    elif n == 7:
        return tan_7(a) / math.sqrt(1 + tan_7(a)**2)
    elif n == 8:
        return tan_8(a) / math.sqrt(1 + tan_8(a)**2)
    elif n == 9:
        return tan_9(a) / math.sqrt(1 + tan_9(a)**2)

def cos(n,a):
    if n == 4:
        return 1 / math.sqrt(1 + tan_4(a)**2)
    elif n == 5:
        return 1 / math.sqrt(1 + tan_5(a)**2)
    elif n == 6:
        return 1 / math.sqrt(1 + tan_6(a)**2)
    elif n == 7:
        return 1 / math.sqrt(1 + tan_7(a)**2)
    elif n == 8:
        return 1 / math.sqrt(1 + tan_8(a)**2)
    elif n == 9:
        return 1 / math.sqrt(1 + tan_9(a)**2)

sin_4_estimations = {a: sin(4,a) for a in random_numbers}
sin_5_estimations = {a: sin(5,a) for a in random_numbers}
sin_6_estimations = {a: sin(6,a) for a in random_numbers}
sin_7_estimations = {a: sin(7,a) for a in random_numbers}
sin_8_estimations = {a: sin(8,a) for a in random_numbers}
sin_9_estimations = {a: sin(9,a) for a in random_numbers}

cos_4_estimations = {a: cos(4,a) for a in random_numbers}
cos_5_estimations = {a: cos(5,a) for a in random_numbers}
cos_6_estimations = {a: cos(6,a) for a in random_numbers}
cos_7_estimations = {a: cos(7,a) for a in random_numbers}
cos_8_estimations = {a: cos(8,a) for a in random_numbers}
cos_9_estimations = {a: cos(9,a) for a in random_numbers}

error_sin_4 = {a: abs(sin_4_estimations[a] - math.sin(a)) for a in random_numbers}
error_sin_5 = {a: abs(sin_5_estimations[a] - math.sin(a)) for a in random_numbers}
error_sin_6 = {a: abs(sin_6_estimations[a] - math.sin(a)) for a in random_numbers}
error_sin_7 = {a: abs(sin_7_estimations[a] - math.sin(a)) for a in random_numbers}
error_sin_8 = {a: abs(sin_8_estimations[a] - math.sin(a)) for a in random_numbers}
error_sin_9 = {a: abs(sin_9_estimations[a] - math.sin(a)) for a in random_numbers}

error_cos_4 = {a: abs(cos_4_estimations[a] - math.cos(a)) for a in random_numbers}
error_cos_5 = {a: abs(cos_5_estimations[a] - math.cos(a)) for a in random_numbers}
error_cos_6 = {a: abs(cos_6_estimations[a] - math.cos(a)) for a in random_numbers}
error_cos_7 = {a: abs(cos_7_estimations[a] - math.cos(a)) for a in random_numbers}
error_cos_8 = {a: abs(cos_8_estimations[a] - math.cos(a)) for a in random_numbers}
error_cos_9 = {a: abs(cos_9_estimations[a] - math.cos(a)) for a in random_numbers}

best_estimation_sin = {}
best_estimation_cos = {}

for a in random_numbers:
    best_estimation = error_sin_4[a]
    best_estimation_sin[a] = "sin_4"
    if error_sin_5[a] < best_estimation:
        best_estimation = error_sin_5[a]
        best_estimation_sin[a] = "sin_5"
    if error_sin_6[a] < best_estimation:
        best_estimation = error_sin_6[a]
        best_estimation_sin[a] = "sin_6"
    if error_sin_7[a] < best_estimation:
        best_estimation = error_sin_7[a]
        best_estimation_sin[a] = "sin_7"
    if error_sin_8[a] < best_estimation:
        best_estimation = error_sin_8[a]
        best_estimation_sin[a] = "sin_8"
    if error_sin_9[a] < best_estimation:
        best_estimation = error_sin_9[a]
        best_estimation_sin[a] = "sin_9"
    best_estimation = error_cos_4[a]
    best_estimation_cos[a] = "cos_4"
    if error_cos_5[a] < best_estimation:
        best_estimation = error_cos_5[a]
        best_estimation_cos[a] = "cos_5"
    if error_cos_6[a] < best_estimation:
        best_estimation = error_cos_6[a]
        best_estimation_cos[a] = "cos_6"
    if error_cos_7[a] < best_estimation:
        best_estimation = error_cos_7[a]
        best_estimation_cos[a] = "cos_7"
    if error_cos_8[a] < best_estimation:
        best_estimation = error_cos_8[a]
        best_estimation_cos[a] = "cos_8"
    if error_cos_9[a] < best_estimation:
        best_estimation = error_cos_9[a]
        best_estimation_cos[a] = "cos_9"

print("\n")
count_cos = Counter(best_estimation_cos.values())
for key, count in sorted(count_cos.items(), key=lambda x: x[0], reverse=True):
    print(f"{key}: {count}")
print("\n")
count_sin = Counter(best_estimation_sin.values())
for key, count in sorted(count_sin.items(), key=lambda x: x[0], reverse=True):
    print(f"{key}: {count}")
print("\n")
count_tan = Counter(best_estimation_tan.values())
for key, count in sorted(count_tan.items(), key=lambda x: x[0], reverse=True):
    print(f"{key}: {count}")