#44 copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn. metrics import accuracy_score
import time
import random
import matplotlib.pyplot as plt

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target   

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=seed, train_size=0.8
    )
#2. 모델
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)    
print("===========", model.__class__.__name__, "===========")
print('acc1 : ', model.score(x_test, y_test))   # acc1 :  0.9722222222222222
print(model.feature_importances_)    


print(" 25% 지점 :  ", np.percentile(model.feature_importances_, 20))
#  25% 지점 : 0.015206812880933285

percentile = np.percentile(model.feature_importances_, 20)
print(type(percentile))  # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25% 이하인놈)을 찾아내자!
for i, fi in enumerate(model.feature_importances_,) : 
    if fi <= percentile : 
        col_name.append(datasets.feature_names[i])
    else : 
        continue                                           # 중지없이 계속 돌아감
print(col_name) # ['sepal width (cm)'] percentile 25보다 낮은 컬럼

x = pd.DataFrame(x, columns=datasets.feature_names)
x = x.drop(columns=col_name)         # 25보다 낮은 컬럼 삭제 

print(x) #[150 rows x 3 columns]

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

# column 이 3개 짜리일 때 성능
model.fit(x_train, y_train)
print('acc2 : ', model.score(x_test, y_test))   # acc2 :  1.0

