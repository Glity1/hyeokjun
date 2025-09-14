import numpy as np
import pandas as pd
import time
import joblib
import warnings
import random
warnings.filterwarnings('ignore')
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())
y = train_csv['Outcome'] 

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

# 최종점수 :  0.6946564885496184


