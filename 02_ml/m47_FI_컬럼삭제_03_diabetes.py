#44 copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
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
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, #stratify=y,
)

#2. 모델
model = XGBRegressor(random_state=seed)

model.fit(x_train, y_train)    
print("===========", model.__class__.__name__, "===========")
print('r2_1 : ', model.score(x_test, y_test))   # acc1 :  0.8364328431288481
print(model.feature_importances_)    


print(" 25% 지점 :  ", np.percentile(model.feature_importances_, 25))
#  25% 지점 : 0.015206812880933285

percentile = np.percentile(model.feature_importances_, 25)
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
    x, y, test_size=0.2, random_state=seed, #stratify=y,
)

# column 이 3개 짜리일 때 성능
model.fit(x_train, y_train)
print('r2_2 : ', model.score(x_test, y_test))   # acc2 :  0.8404782483873166
