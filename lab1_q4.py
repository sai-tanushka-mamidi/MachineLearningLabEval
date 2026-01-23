#Write a program that accepts a matrix as input and returns its transpose
rows = int(input("Rows: "))
cols = int(input("Columns: "))
matrix = []
for i in range(rows):
    row = []
    for j in range(cols):
        row.append(int(input(f"Enter element [{i+1}][{j+1}]: ")))
    matrix.append(row)
transpose = [[matrix[i][j] for i in range(rows)] for j in range(cols)]
print("Transpose of the matrix:")
for row in transpose:
    print(row)
