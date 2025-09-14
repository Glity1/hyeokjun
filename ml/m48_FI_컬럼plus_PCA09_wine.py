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
from sklearn.decomposition import PCA

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
print(col_name) # ['alcalinity_of_ash', 'nonflavanoid_phenols', 'proanthocyanins'] // percentile 25보다 낮은 컬럼

# exit()

x_f = pd.DataFrame(x, columns=datasets.feature_names)
x1 = x_f.drop(columns=col_name)         # 25보다 낮은 컬럼 삭제 
x2 = x_f[['alcalinity_of_ash', 'nonflavanoid_phenols', 'proanthocyanins']]
print(x2) #[178 rows x 3 columns]

x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, test_size=0.2, random_state=seed, #stratify=y,
)

pca = PCA(n_components=2)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)


x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)

model.fit(x_train, y_train)
print('FI_Drop + PCA : ', model.score(x_test, y_test)) 

# FI_Drop + PCA :  0.9722222222222222