import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
data = pd.read_excel("speaker_clarity_big.xlsx")
person1 = data[["clarity_score", "speech_rate"]].iloc[0].values
person2 = data[["clarity_score", "speech_rate"]].iloc[1].values
def easy_distance(a, b, p):
    total = 0

    for i in range(len(a)):
        diff = a[i] - b[i]        
        if diff < 0:
            diff = -diff        

        total = total + (diff ** p)
        root = total ** (1 / p)      
    return root
p_values = [1, 2, 3]

for p in p_values:
    my_d = easy_distance(person1, person2, p)
    lib_d = minkowski(person1, person2, p)

    print("\np value:", p)
    print("My distance:", my_d)
    print("Library distance:", lib_d)
