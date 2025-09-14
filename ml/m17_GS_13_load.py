from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import joblib
import pandas as pd
import numpy as np

#1. 데이터
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']             

le = LabelEncoder() # 문자열을 숫자로 현재 y에 들어간 train_csv의 target은 문자열임
y = le.fit_transform(y)  

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

path = './_save/m15_cv_results/'
model = joblib.load(path + 'm16_otto_best_model.joblib')

#4. 평가, 예측
print('- mode.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)                                      # 두 predict 두개중에 원하는거 쓰면된다
print('- accuracy_score : ', accuracy_score(y_test, y_pred))


# - 최적의 파라미터 :  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
# - best_score :  0.95
# - mode.score :  0.9666666666666667
# - accuracy_score :  0.9666666666666667
# - best_accuracy_score :  0.9666666666666667
# - 걸린시간 :  4.26 초
