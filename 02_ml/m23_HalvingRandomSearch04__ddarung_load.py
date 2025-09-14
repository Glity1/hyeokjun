# m15_00 copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)   
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55
)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=503)

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12], 'learning_rate': [0.1, 0.01, 0.001]},    # 18
    {'max_depth': [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]},                            # 12
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}                       # 12
]

path = './_save/m15_cv_results/'
model = joblib.load(path + 'm18_04_best_model2.joblib')

print('- mode.score : ', model.score(x_test, y_test))
 
y_pred = model.predict(x_test)                                      # 두 predict 두개중에 원하는거 쓰면된다

print("R2 Score :", r2_score(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))

# - mode.score :  0.7576403352472629
# R2 Score : 0.7576403352472629
# MSE : 1634.167306673649


















