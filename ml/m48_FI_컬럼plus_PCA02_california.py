#44 copy
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
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
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, #stratify=y,
)

#2. 모델
model = XGBRegressor(random_state=seed)

model.fit(x_train, y_train)    
print("===========", model.__class__.__name__, "===========")
print('r2_1 : ', model.score(x_test, y_test))   # acc1 :  0.8364328431288481
# print(model.feature_importances_)    


print(" 25% 지점 :  ", np.percentile(model.feature_importances_, 25))
#  25% 지점 :   0.0410702358931303

percentile = np.percentile(model.feature_importances_, 25)
# print(type(percentile))  # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25% 이하인놈)을 찾아내자!
for i, fi in enumerate(model.feature_importances_,) : 
    if fi <= percentile : 
        col_name.append(datasets.feature_names[i])
    else : 
        continue                                           # 중지없이 계속 돌아감
print(col_name) # ['AveBedrms', 'Population'] // percentile 25보다 낮은 컬럼

x_f = pd.DataFrame(x, columns=datasets.feature_names)
x1 = x_f.drop(columns=col_name)         # 25보다 낮은 컬럼 삭제 
x2 = x_f[['AveBedrms', 'Population']]
print(x2) #[20640 rows x 2 columns]

x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, test_size=0.2, random_state=seed, #stratify=y,
)
# print(x1_train.shape, x1_test.shape) # (16512, 6) (4128, 6)
# print(x2_train.shape, x2_test.shape) # (16512, 2) (4128, 2)
# print(y_train.shape, y_test.shape)   # (16512,) (4128,)

pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)

# print(x2_train.shape, x2_test.shape) # (16512, 1) (4128, 1)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)

# print(x_train.shape, x_test.shape) # (16512, 7) (4128, 7)

model.fit(x_train, y_train)
print('FI_Drop + PCA : ', model.score(x_test, y_test))   
# r2_2 :  0.8404782483873166 (dorp만)
# FI_Drop + PCA :  0.8370725229374802 (drop한거 + drop을 PCA한거 )                                                     