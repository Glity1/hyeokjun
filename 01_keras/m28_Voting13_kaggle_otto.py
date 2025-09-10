import numpy as np
import pandas as pd
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
import random
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingClassifier
from catboost import CatBoostClassifier

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']             

le = LabelEncoder() 
y = le.fit_transform(y)

n_classes = len(le.classes_)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8,  stratify=y
    )

#2. 모델
xgb = XGBClassifier(
    objective='multi:softmax', # 다중 클래스 분류를 위한 objective  
    num_class= n_classes,     # 클래스 개수 명시
    use_label_encoder=False, # DeprecationWarning 방지 및 최신 사용법 권장
    eval_metric='mlogloss',  # 다중 분류 평가 지표
    n_jobs=-1                # 모든 코어 사용/
)
lg = LGBMClassifier(
    objective='multiclass',  # 다중 클래스 분류를 위한 objective
    num_class= n_classes,     # 클래스 개수 명시
    n_jobs=-1                # 모든 코어 사용
)
cat = CatBoostClassifier(
    loss_function='MultiClass', # 다중 클래스 분류를 위한 손실 함수
    classes_count= n_classes,    # 클래스 개수 명시
    verbose=0,                  # 훈련 중 출력되는 메시지 억제
)

model = VotingClassifier(
    estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
    # voting='soft',
    weights=[2,1,1],
    voting='hard', 
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

# 최종점수  : 0.81439883645766  # softvoting
# 최종점수 :  0.9736842105263158  # hardvoting



