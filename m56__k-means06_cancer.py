#47-0 copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
import time
import random
import matplotlib.pyplot as plt

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(y_train[:10])
# [0 1 1 0 1 1 1 1 1 1]
# exit()

#2. 모델
model = KMeans(n_clusters=2, init='k-means++',
               n_init=10, random_state=337)

y_train_pred = model.fit_predict(x_train)

cluster_0_indices = np.where(y_train_pred == 0)[0]
cluster_1_indices = np.where(y_train_pred == 1)[0]

label_0_in_cluster_0 = np.sum(y_train[cluster_0_indices] == 0)
label_1_in_cluster_0 = np.sum(y_train[cluster_0_indices] == 1)

label_0_in_cluster_1 = np.sum(y_train[cluster_1_indices] == 0)
label_1_in_cluster_1 = np.sum(y_train[cluster_1_indices] == 1)

print("Cluster 0:", label_0_in_cluster_0, "vs", label_1_in_cluster_0)
print("Cluster 1:", label_0_in_cluster_1, "vs", label_1_in_cluster_1)

# print(y_train_pred[:10])
# print(y_train[:10])
# [0 1 1 1 1 1 1 1 1 1]
# [0 1 1 0 1 1 1 1 1 1]

# exit()

print("===========", model.__class__.__name__, "===========")
print('acc : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(" acc : ", acc)
f1 = f1_score(y_test ,y_pred)
print(" f1 : ", f1)

# =========== KNeighborsClassifier ===========
# acc :  0.9473684210526315
#  acc :  0.9473684210526315
#  f1 :  0.958904109589041