import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55
)

#2. 모델
# model = DecisionTreeRegressor()

model = BaggingRegressor(DecisionTreeRegressor(),
                         n_estimators=100,
                         n_jobs=-1,
                         random_state=seed,
                        #  bootstrap=True,
                         )
# model = RandomForestRegressor(random_state=seed)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

# 최종점수 :  0.5053728888697607


