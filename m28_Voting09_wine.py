import numpy as np
import pandas as pd
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
import random
from bayes_opt import BayesianOptimization
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target   

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

scaler = RobustScaler()
x = scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
    # voting='soft',
    voting='hard',   
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

# 최종점수 :  1.0  # softvoting
# 최종점수 :   # hardvoting

