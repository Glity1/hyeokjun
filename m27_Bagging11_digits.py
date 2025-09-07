import numpy as np
import pandas as pd
import time
import joblib
import warnings
import random
warnings.filterwarnings('ignore')
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target    

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

#2. 모델
# model = DecisionTreeRegressor()

model = BaggingClassifier(DecisionTreeClassifier(),
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

# 최종점수 :  0.9583333333333334

