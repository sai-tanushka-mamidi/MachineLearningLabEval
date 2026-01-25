import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import minkowski

# A7: Train a kNN classifier (k = 3)
data = pd.read_excel("speaker_clarity_big.xlsx")
data = data[(data["label"] == "Clear") | (data["label"] == "Unclear")]

x = data[["clarity_score", "speech_rate"]]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# A8: Test the accuracy of the kNN
acc = knn.score(x_test, y_test)
print("A8 Accuracy:", acc)

# A9: Predict test data and one test vector
y_pred = knn.predict(x_test)
print("A9 First 5 predictions:", y_pred[:5])

one_person = x_test.iloc[0].values.reshape(1, -1)
print("A9 One person prediction:", knn.predict(one_person))

# A10: Own Minkowski + Own kNN
def easy_distance(a, b, p):
    total = 0
    for i in range(len(a)):
        d = a[i] - b[i]
        if d < 0:
            d = -d
        total = total + (d * d) if p == 2 else total + (d ** p)
    return total ** (1 / p)

p1 = x_train.iloc[0].values
p2 = x_train.iloc[1].values

for p in [1, 2]:
    my_d = easy_distance(p1, p2, p)
    lib_d = minkowski(p1, p2, p)
    print("A10 p =", p, "My:", my_d, "Lib:", lib_d)

def my_knn(train_x, train_y, test_x, k):
    d_list = []
    for i in range(len(train_x)):
        diff = train_x.iloc[i].values - test_x
        sq = diff * diff
        dist = sq.sum() ** 0.5
        d_list.append((dist, train_y.iloc[i]))

    d_list.sort(key=lambda x: x[0])
    top = d_list[:k]

    votes = {}
    for d, label in top:
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)

print("A10 My kNN:", my_knn(x_train, y_train, x_test.iloc[0].values, 3))
print("A10 Lib kNN:", knn.predict(x_test.iloc[0].values.reshape(1, -1)))

# A11: Vary k from 1 to 11 and plot accuracy
k_list = []
a_list = []

for k in range(1, 12):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(x_train, y_train)
    a = m.score(x_test, y_test)
    k_list.append(k)
    a_list.append(a)

plt.plot(k_list, a_list)
plt.title("A11 Accuracy vs k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()

# A12: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("A12 Confusion Matrix:\n", cm)

# A13: Own confusion + metrics
def my_confusion(y_true, y_pred):
    TP = FP = TN = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == "Clear" and y_pred[i] == "Clear":
            TP = TP + 1
        elif y_true[i] == "Clear" and y_pred[i] == "Unclear":
            FN = FN + 1
        elif y_true[i] == "Unclear" and y_pred[i] == "Clear":
            FP = FP + 1
        else:
            TN = TN + 1
    return TP, FP, TN, FN

def my_metrics(TP, FP, TN, FN):
    total = TP + FP + TN + FN
    acc = (TP + TN) / total
    prec = TP / (TP + FP) if (TP + FP) != 0 else 0
    rec = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
    return acc, prec, rec, f1

TP, FP, TN, FN = my_confusion(y_test.values, y_pred)
a, p, r, f = my_metrics(TP, FP, TN, FN)

print("A13 Accuracy:", a)
print("A13 Precision:", p)
print("A13 Recall:", r)
print("A13 F1:", f)

# A14: Matrix inversion method
y_num = y_train.map({"Clear": 1, "Unclear": 0}).values

bias = np.ones(len(x_train))
X = np.c_[bias, x_train.values]

XT = X.T
W = np.linalg.inv(XT @ X) @ XT @ y_num

bias_test = np.ones(len(x_test))
X_test_new = np.c_[bias_test, x_test.values]

y_out = X_test_new @ W

y_class = []
for v in y_out:
    if v >= 0.5:
        y_class.append("Clear")
    else:
        y_class.append("Unclear")

correct = 0
for i in range(len(y_class)):
    if y_class[i] == y_test.values[i]:
        correct = correct + 1

print("A14 Matrix inversion accuracy:", correct / len(y_class))
