#Q3
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
data = pd.read_excel("Lab _Session _Data.xlsx",
price_data = data.iloc[:, 3]
change_data = data.iloc[:, 8]
date_data = pd.to_datetime(data.iloc[:, 0])
population_mean = np.mean(price_data)
population_variance = np.var(price_data)

print("Mean (NumPy):", population_mean)
print("Variance (NumPy):", population_variance)
def calculate_mean(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)

def calculate_variance(values):
    m = calculate_mean(values)
    total = 0
    for v in values:
        total += (v - m) ** 2
    return total / len(values)

print("Mean (Custom):", calculate_mean(price_data))
print("Variance (Custom):", calculate_variance(price_data))
def average_time(function, values):
    total_time = 0
    for i in range(10):
        start = time.time()
        function(values)
        total_time += time.time() - start
    return total_time / 10

print("NumPy Mean Time:", average_time(np.mean, price_data))
print("Custom Mean Time:", average_time(calculate_mean, price_data))

data["Day"] = date_data.dt.day_name()
data["Month"] = date_data.dt.month
wednesday_prices = data[data["Day"] == "Wednesday"].iloc[:, 3]
print("Wednesday Mean:", np.mean(wednesday_prices))
print("Population Mean:", population_mean)
april_prices = data[data["Month"] == 4].iloc[:, 3]
print("April Mean:", np.mean(april_prices))
loss_days = list(filter(lambda x: x < 0, change_data))
loss_probability = len(loss_days) / len(change_data)
print("Probability of Loss:", loss_probability)

wednesday_changes = data[data["Day"] == "Wednesday"].iloc[:, 8].dropna()
profit_days = wednesday_changes[wednesday_changes > 0]

profit_probability = len(profit_days) / len(wednesday_changes)
print("Profit Probability on Wednesday:", profit_probability)

# ------------------------------------------------
# Scatter Plot
# ------------------------------------------------
plt.scatter(data["Day"], data.iloc[:, 8])
plt.xlabel("Day of the Week")
plt.ylabel("Change Percentage (Chg%)")
plt.title("Chg% vs Day of Week")
plt.xticks(rotation=45)
plt.show()
