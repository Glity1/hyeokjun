# 참고 : pseudo Labeling 

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222, #stratify=y)
    ) 

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)/
# x_test = scaler.transform(x_test)

#2-1. 모델
xgb = XGBClassifier()
# rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)
lg = LGBMClassifier(verbose=0)

# models = [xgb, rf, cat, lg]
models = [xgb, cat, lg]
# models = [rf, cat, lg]


train_list = []
test_list = []

for model in models : 
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)
    test_list.append(y_test_pred)
    
    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__                         # 모델 이름이 나옴
    print('{0} ACC : {1:.4f}'.format(class_name, score))

# XGBClassifier ACC : 0.9737
# RandomForestClassifier ACC : 0.9649
# CatBoostClassifier ACC : 0.9649
# LGBMClassifier ACC : 0.9649

x_train_new = np.array(train_list).T
# print(x_train_new)
print(x_train_new.shape) # (455, 4)

x_test_new = np.array(test_list).T 
print(x_test_new.shape)  # (114, 4)

#2-2. 모델

# model2 = XGBClassifier(
#     n_estimators=100,
#     learning_rate=0.05,
#     max_depth=3,
#     min_child_weight=1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     gamma=0,
#     reg_alpha=0.01,
#     reg_lambda=1.0,
#     objective='binary:logistic',
#     eval_metric='logloss',
#     use_label_encoder=False,
#     random_state=222,
#     verbosity=0
# )

model2 = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=1.0,
    random_state=222,
    verbose=-1
)

# model2 = CatBoostClassifier(
#     iterations=100,
#     learning_rate=0.05,
#     depth=3,
#     l2_leaf_reg=3,
#     subsample=0.8,
#     random_state=222,
#     verbose=0
# )

# model2 = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=3,
#     min_samples_leaf=4,
#     min_samples_split=5,
#     max_features='sqrt',
#     bootstrap=True,
#     random_state=222,
#     verbose=0
# )

model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred2)
print("스태킹 결과 : ", score2)  # 스태킹 결과 :  0.888961038961039