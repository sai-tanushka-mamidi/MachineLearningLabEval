#Write a program that accepts two matrices A and B as input and returns their product AB. 
#Check if A & B are multipliable; if not, return error message.

import numpy as np
rA = int(input("Rows of A: "))
cA = int(input("Cols of A: "))
rB = int(input("Rows of B: "))
cB = int(input("Cols of B: "))
if cA == rB:
    print("Enter A:")
    A = np.array([list(map(int, input().split())) for _ in range(rA)])
    print("Enter B:")
    B = np.array([list(map(int, input().split())) for _ in range(rB)])
    print("Product of A and B:")
    print(np.dot(A, B))
else:
    print("Multiplication not possible")
