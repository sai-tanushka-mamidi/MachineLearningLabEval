import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel("speaker_clarity_big.xlsx")
person1 = data[["clarity_score", "speech_rate"]].iloc[0].values
person2 = data[["clarity_score", "speech_rate"]].iloc[1].values
def easy_distance(a, b, p):
    total = 0
     for i in range(len(a)):
        diff = a[i]-b[i]        
        if diff < 0:
            diff=-diff         
            total=total+(diff*diff)
            if p == 2 else total + (diff ** p)

            root = total ** (1 / p)      
    return root
p_list = []
d_list = []
for p in range(1, 11):
    d = easy_distance(person1, person2, p)
    p_list.append(p)
    d_list.append(d)
plt.plot(p_list, d_list, marker="o")
plt.title("Distance vs p")
plt.xlabel("p value")
plt.ylabel("Distance")
plt.show()
print("Person 1:", person1)
print("Person 2:", person2)
print("p values:", p_list)
print("Distances:", d_list)

