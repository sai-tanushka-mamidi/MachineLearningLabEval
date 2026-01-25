import numpy as np
import pandas as pd
data = pd.read_excel("speaker_clarity_big.xlsx")
score = data["clarity_score"]
speed = data["speech_rate"]
group = data["label"]
box1 = []   
box2 = []   
for i in range(len(group)):
    if group[i] == "Clear":
        box1.append([score[i], speed[i]])
    else:
        box2.append([score[i], speed[i]])

box1 = np.array(box1)
box2 = np.array(box2)
avg1 = np.mean(box1, axis=0)
avg2 = np.mean(box2, axis=0)
spread1 = np.std(box1, axis=0)
spread2 = np.std(box2, axis=0)
distance = np.linalg.norm(avg1 - avg2)
print("Box 1 average:", avg1)
print("Box 2 average:", avg2)
print("Box 1 spread:", spread1)
print("Box 2 spread:", spread2)
print("Distance between boxes:", distance)
