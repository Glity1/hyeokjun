from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import time
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = './_data/kaggle/santander/'                                          
train_csv = pd.read_csv(path + 'train.csv', index_col=0)                    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)                      

x = train_csv.drop(['target'], axis=1)                                     
y = train_csv['target']                                                    

x_train, x_test, y_train, y_test = train_test_split(                       
    x, y, test_size=0.2, random_state=74, stratify=y
)

scaler = RobustScaler()
x = scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

path = './_save/m15_cv_results/'
model = joblib.load(path + 'm16_santander_best_model.joblib')

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
