import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel("speaker_clarity_big.xlsx")
scores = data["clarity_score"]
counts, bins = np.histogram(scores, bins=5)
mean_value = np.mean(scores)
variance_value = np.var(scores)
plt.hist(scores, bins=5)
plt.title("Histogram of Clarity Score")
plt.xlabel("Clarity Score")
plt.ylabel("Count")
plt.show()
print("Histogram counts:", counts)
print("Histogram bins:", bins)
print("Mean:", mean_value)
print("Variance:", variance_value)
