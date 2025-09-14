# m14_00 copy
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

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12], 'learning_rate': [0.1, 0.01, 0.001]},    # 18
    {'max_depth': [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]},                            # 12
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}                       # 12
]

#2. 모델
xgb = XGBClassifier(tree_method='hist', device='cuda', random_state=42)
model = GridSearchCV(xgb, parameters, cv=kfold,   # 42 * 5 = 210
                     verbose=2,
                     n_jobs=-1,
                     refit=True,    # 1번
                     )  # 총 210번 + 1번 = 211번

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start

print('\n\n- 최적의 매개변수 : ', model.best_estimator_)            # refit=false 상태일 때 print다 안됨
print('- 최적의 파라미터 : ', model.best_params_)

#4. 평가, 예측
print('- best_score : ', model.best_score_)
print('- mode.score : ', model.score(x_test, y_test))
 
y_pred = model.predict(x_test)                                      # 두 predict 두개중에 원하는거 쓰면된다
print('- accuracy_score : ', accuracy_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('- best_accuracy_score : ', accuracy_score(y_test, y_pred_best))


print('- 걸린시간 : ', round(end, 2), '초\n\n')

# - 최적의 파라미터 :  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
# - best_score :  0.95
# - mode.score :  0.9666666666666667
# - accuracy_score :  0.9666666666666667
# - best_accuracy_score :  0.9666666666666667
# - 걸린시간 :  4.26 초


path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_, path + 'm16_santander_best_model.joblib' )








