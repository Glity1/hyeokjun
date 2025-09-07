#44 copy
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
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
datasets = fetch_covtype()
x = datasets.data
y = datasets.target   

le = LabelEncoder()
y = le.fit_transform(y)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, #stratify=y
)
#2. 모델
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)    
print("===========", model.__class__.__name__, "===========")
print('acc1 : ', model.score(x_test, y_test))   # acc1 :  0.8708294966567128
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
print(col_name) # ['Slope', 'Hillshade_3pm', 'Soil_Type_0', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_25', 'Soil_Type_27', 'Soil_Type_33', 'Soil_Type_35'] // percentile 25보다 낮은 컬럼

# exit()
x_f = pd.DataFrame(x, columns=datasets.feature_names)
x1 = x_f.drop(columns=col_name)         # 25보다 낮은 컬럼 삭제 
x2 = x_f[['Slope', 'Hillshade_3pm', 'Soil_Type_0', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_25', 'Soil_Type_27', 'Soil_Type_33', 'Soil_Type_35']]
print(x2) #[581012 rows x 14 columns]

x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, test_size=0.2, random_state=seed, #stratify=y,
)

pca = PCA(n_components=5)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)


x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)

model.fit(x_train, y_train)
print('FI_Drop + PCA : ', model.score(x_test, y_test))

# FI_Drop + PCA :  0.8721375523867714