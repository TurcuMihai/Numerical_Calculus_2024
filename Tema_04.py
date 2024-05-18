
# A = {}
# with open('a.txt', 'r') as file:
#     number = int(file.readline())
#     for line in file:
#         line = line.strip().split(', ')
#         if int(line[1]) not in A.keys():
#             A[int(line[1])] = {}
#         if int(line[2]) not in A[int(line[1])].keys():
#             A[int(line[1])][int(line[2])] = float(line[0])
#         else:
#             A[int(line[1])][int(line[2])] += float(line[0])

A_1 = {}
with open('a_1.txt', 'r') as file:
    number = int(file.readline())
    for line in file:
        line = line.replace(" ", "").replace("\n","").split(',')
        if int(line[1]) not in A_1.keys():
            A_1[int(line[1])] = {}
        if int(line[2]) not in A_1[int(line[1])].keys():
            A_1[int(line[1])][int(line[2])] = float(line[0])
        else:
            A_1[int(line[1])][int(line[2])] += float(line[0])

b_1 = []
with open('b_1.txt', 'r') as file:
    for line in file:
        b_1.append(float(line.replace("\n", "")))


A_2 = {}
with open('a_2.txt', 'r') as file:
    number = int(file.readline())
    for line in file:
        line = line.replace(" ", "").replace("\n","").split(',')
        if int(line[1]) not in A_2.keys():
            A_2[int(line[1])] = {}
        if int(line[2]) not in A_2[int(line[1])].keys():
            A_2[int(line[1])][int(line[2])] = float(line[0])
        else:
            A_2[int(line[1])][int(line[2])] += float(line[0])

b_2 = []
with open('b_2.txt', 'r') as file:
    for line in file:
        b_2.append(float(line.replace("\n", "")))




A_3 = {}
with open('a_3.txt', 'r') as file:
    number = int(file.readline())
    for line in file:
        line = line.replace(" ", "").replace("\n","").split(',')
        if int(line[1]) not in A_3.keys():
            A_3[int(line[1])] = {}
        if int(line[2]) not in A_3[int(line[1])].keys():
            A_3[int(line[1])][int(line[2])] = float(line[0])
        else:
            A_3[int(line[1])][int(line[2])] += float(line[0])

b_3 = []
with open('b_3.txt', 'r') as file:
    for line in file:
        b_3.append(float(line.replace("\n", "")))
        

A_4 = {}
with open('a_4.txt', 'r') as file:
    number = int(file.readline())
    for line in file:
        line = line.replace(" ", "").replace("\n","").split(',')
        if int(line[1]) not in A_4.keys():
            A_4[int(line[1])] = {}
        if int(line[2]) not in A_4[int(line[1])].keys():
            A_4[int(line[1])][int(line[2])] = float(line[0])
        else:
            A_4[int(line[1])][int(line[2])] += float(line[0])

b_4 = []
with open('b_4.txt', 'r') as file:
    for line in file:
        b_4.append(float(line.replace("\n", "")))



A_5 = {}
with open('a_5.txt', 'r') as file:
    number = int(file.readline())
    for line in file:
        line = line.replace(" ", "").replace("\n","").split(',')
        if int(line[1]) not in A_5.keys():
            A_5[int(line[1])] = {}
        if int(line[2]) not in A_5[int(line[1])].keys():
            A_5[int(line[1])][int(line[2])] = float(line[0])
        else:
            A_5[int(line[1])][int(line[2])] += float(line[0])

b_5 = []
with open('b_5.txt', 'r') as file:
    for line in file:
        b_5.append(float(line.replace("\n", "")))
    
