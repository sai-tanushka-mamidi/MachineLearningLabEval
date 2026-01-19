#Q4
import pandas as pd
import numpy as np

# Load the Excel file
data = pd.read_excel("Lab _Session _Data.xlsx",
                     sheet_name="thyroid0387_UCI")

# Print dataset information
print("Number of rows and columns:")
print(data.shape)

print("\nFirst 5 records:")
print(data.head())
print("\nAttribute Datatypes:")
for col in data.columns:
    print(col, ":", data[col].dtype)
categorical = []
numeric = []
for col in data.columns:
    if data[col].dtype == "object":
        categorical.append(col)
    else:
        numeric.append(col)

print("\nCategorical Attributes:", categorical)
print("Numeric Attributes:", numeric)
print("\nEncoding Scheme:")
for col in categorical:
    print(col, "-> One-Hot Encoding (Nominal data)")
print("\nRange of Numeric Attributes:")
for col in numeric:
    print(col, "Min =", data[col].min(),
          "Max =", data[col].max())

print("\nMissing Values in each attribute:")
for col in data.columns:
    print(col, ":", data[col].isnull().sum())
print("\nOutlier Count:")
for col in numeric:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = data[(data[col] < lower) | (data[col] > upper)]
    print(col, "Outliers =", len(outliers))

print("\nMean and Variance:")
for col in numeric:
    print(col,
          "Mean =", round(data[col].mean(), 3),
          "Variance =", round(data[col].var(), 3),
          "Std Dev =", round(data[col].std(), 3))
