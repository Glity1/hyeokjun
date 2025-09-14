#47-0 copy
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import time
import random
import matplotlib.pyplot as plt

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())
y = train_csv['Outcome'] 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)   
 
print("===========", model.__class__.__name__, "===========")
print('acc : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(" acc : ", acc)
f1 = f1_score(y_test ,y_pred)
print(" f1 : ", f1)

# =========== KNeighborsClassifier ===========
# acc :  0.7175572519083969
# acc :  0.7175572519083969
# f1 :  0.6021505376344086