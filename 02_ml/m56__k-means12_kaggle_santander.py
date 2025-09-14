#47-0 copy
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
import time
import random
import matplotlib.pyplot as plt

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './_data/kaggle/santander/'                                          
train_csv = pd.read_csv(path + 'train.csv', index_col=0)                    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)                      

x = train_csv.drop(['target'], axis=1)                                     
y = train_csv['target']                                                    

x_train, x_test, y_train, y_test = train_test_split(                       
    x, y, test_size=0.2, random_state=seed, stratify=y
)

scaler = RobustScaler()
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
f1 = f1_score(y_test, y_pred, average='macro')
print(" f1 : ", f1)

# =========== KNeighborsClassifier ===========
# acc :  0.899425
#  acc :  0.899425
#  f1 :  0.47402013623630246