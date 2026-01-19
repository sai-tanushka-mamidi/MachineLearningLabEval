#Q1
import pandas as pd
import numpy as np
def calculate_rank_and_cost(Lab_Session_Data):
    data = pd.read_excel(Lab_Session_Data.xlsx)
    X = data.iloc[:, 1:4].to_numpy()
    y = data.iloc[:, 4].to_numpy()
    matrix_rank = np.linalg.matrix_rank(X)
    print(f"Rank of the feature matrix: {matrix_rank}")
    unit_cost = np.matmul(np.linalg.pinv(X), y)

    return unit_cost
costs = calculate_rank_and_cost("Lab Session Data.xlsx")

print("Unit cost of candies:", costs[0])
print("Unit cost of mangoes:", costs[1])
print("Unit cost of milk:", costs[2])
