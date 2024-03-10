import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# creates a set of if-then clauses, using clustering methods
def FSMGeneralization(X, y, eps):
    if eps == 0:
        # no generalization
        print("Generalization distance eps: {}.".format(eps))
        print("Number of if-then clauses: {}.".format(len(y)))
        print("Training accuracy: {:.2f}%.".format(100))
        return
    
    eps = eps
    db = DBSCAN(eps=eps, min_samples=1) 
    db.fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # group labels
    labels = np.hstack([labels.reshape(-1,1), np.arange(len(X)).reshape(-1,1)])
    groups = np.split(labels[:,1], np.unique(labels[:, 0], return_index=True)[1][1:])
    n_correct = len(X)
    clusters_label = []
    for g in groups:
        if len(g) > 1:
            # majority vote
            values, counts = np.unique(y[g], return_counts=True)
            clusters_label.append(values[np.argmax(counts)])
            if len(values) > 1:
                n_correct -= counts.min()

    accuracy = n_correct / len(X)
    # if len(set(clusters_label)) == 1:
    #     n_clusters_ = 1

    print("Generalization distance eps: {}.".format(eps))
    print("Number of if-then clauses: {}.".format(n_clusters_))
    print("Training accuracy: {:.2f}%.".format(accuracy * 100))

# Titanic Dataset
# read in data, drop columns, convert categorical data to one-hot encoding
df = pd.read_csv("titanic.csv")
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
df = pd.get_dummies(df, columns=["Pclass", "Sex", "Embarked"])
df.fillna(df.median(), inplace=True)

y = df["Survived"].to_numpy()
X = df.to_numpy()

print("Titanic Dataset: Total number of instances: {}.".format(len(y)))
print()
FSMGeneralization(X, y, eps=0)
print("---------------------------------------------")
FSMGeneralization(X, y, eps=0.1)
print("---------------------------------------------")
FSMGeneralization(X, y, eps=0.5)
print("---------------------------------------------")
FSMGeneralization(X, y, eps=1000)
print("=================================")

## Ionosphere Dataset
# read in data
df = pd.read_csv("ionosphere.csv", header=None)
df.iloc[:, -1] = df.iloc[:, -1].map({'g': 1, 'b': 0})

y = df[df.columns[-1]].to_numpy()
X = df.to_numpy()

print("Ionosphere Dataset: Total number of instances: {}.".format(len(y)))
print()
FSMGeneralization(X, y, eps=0)
print("---------------------------------------------")
FSMGeneralization(X, y, eps=0.1)
print("---------------------------------------------")
FSMGeneralization(X, y, eps=0.5)
print("---------------------------------------------")
FSMGeneralization(X, y, eps=1000)
print("=================================")

## Heart Disease Dataset
# read in data, convert categorical data to one-hot encoding
df = pd.read_csv("heart_disease.csv")
df = pd.get_dummies(df, columns=["cp", "restecg"])

y = df["target"].to_numpy()
X = df.to_numpy()

print("Heart Disease Dataset: Total number of instances: {}.".format(len(y)))
print()
FSMGeneralization(X, y, eps=0)
print("---------------------------------------------")
FSMGeneralization(X, y, eps=0.1)
print("---------------------------------------------")
FSMGeneralization(X, y, eps=0.5)
print("---------------------------------------------")
FSMGeneralization(X, y, eps=1000)
print("=================================")


## Random Dataset
Ns = [100, 200, 1000] 
Ds = [10, 100]
EPSs = [10, 1000]


for N in Ns:
    for D in Ds:
        X = np.random.uniform(-10, 10, size=(N, D))
        y = np.random.choice([0, 1], size=(N,))
        print("Number of instances: {}. Number of input columns: {}".format(N, D))
        for eps in EPSs:
            FSMGeneralization(X, y, eps=eps)
        print()