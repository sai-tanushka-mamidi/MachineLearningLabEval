#Q5
import numpy as np
import pandas as pd
v1 = df.iloc[0]
v2 = df.iloc[1]
binary = [
    col for col in df.columns
    if set(df[col].dropna().unique()).issubset({0, 1})
]
b1 = v1[binary]
b2 = v2[binary]
f11 = np.sum((b1 == 1) & (b2 == 1))
f10 = np.sum((b1 == 1) & (b2 == 0))
f01 = np.sum((b1 == 0) & (b2 == 1))
f00 = np.sum((b1 == 0) & (b2 == 0))
if (f11 + f10 + f01) == 0:
    jaccard = 0
else:
    jaccard = f11 / (f11 + f10 + f01)
similarity_coefficient = (f11 + f00) / (f11 + f10 + f01 + f00)
print("Jaccard Similarity:", jaccard)
print("Simple Matching Coefficient:", similarity_coefficient)
