#49_06 copy
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn. metrics import r2_score, accuracy_score
import time
import random
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

seed = 444
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
feature_names = train_csv.columns

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())
y = train_csv['Outcome'] 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=seed, train_size=0.8
    )

#2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = 'logloss',
    data_name = 'validation_0',
    # save_best = True,
)

model = XGBClassifier(
    n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,
    eval_metric = 'logloss',  # 다중분류 : mlogloss, merror, 
                              # 이진분류 : logloss, error
                              # 회귀 : rmse, mae, mrsle
                              # 2.1.1 버전 이후로 fit에서 모델로 위치이동.
    callbacks = [es],
    random_state=seed)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 0,
          )    

print('acc : ', model.score(x_test, y_test))  

score_dict = model.get_booster().get_score(importance_type='gain')  
score_list = np.array(list(score_dict.values()))
print(score_list)
exit()


thresholds = np.sort(score_list)  
print(thresholds)

delete_columns = []
max_acc = 0

for i in thresholds : 
    selection = SelectFromModel(model, threshold=i, prefit=False)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    if select_x_train.shape[1] == 0:
        print(f"Thresh={i:.3f},  No features selected → Skipping")
        continue

    select_model = XGBClassifier(
        n_estimators = 500,
        max_depth = 6,
        gamma = 0,
        min_child_weight = 0,
        subsample = 0.4,
        reg_alpha = 0,
        reg_lambda = 1,
        eval_metric = 'logloss',  # 다중분류 : mlogloss, merror, 이진분류 : logloss, error
                                # 2.1.1 버전 이후로 fit에서 모델로 위치이동.
        # early_stopping_rounds=30,
        # callbacks = [es],
        random_state=seed
        )
    
    select_model.fit(select_x_train, y_train,
          eval_set = [(select_x_test, y_test)],
          verbose = 0,
          )    
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)     
    print('Thresh=%.3f, n=%dscore, ACC: %.4f%%' %(i, select_x_train.shape[1], score*100))
    print("==============================================")

    removed_features = [feature_names[j] for j in range(len(score_list)) if score_list[j] < i]

    
    if score >= max_acc:
        max_acc = score
        delete_columns = removed_features.copy()
        
print(' 가장 높은 정확도: %.4f%%' % (max_acc * 100))
print(' 이때 삭제할 컬럼들 (%d개):' % len(delete_columns), delete_columns)   














