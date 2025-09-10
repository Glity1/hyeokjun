# import numpy as np
# import pandas as pd
# import time
# import joblib
# import warnings
# warnings.filterwarnings('ignore')
# import random
# from bayes_opt import BayesianOptimization
# from sklearn.datasets import load_digits
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.metrics import r2_score, accuracy_score
# from sklearn.preprocessing import LabelEncoder, RobustScaler
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingClassifier
# from lightgbm import LGBMClassifier, early_stopping
# from catboost import CatBoostClassifier
# from sklearn.base import ClassifierMixin

# seed = 222
# random.seed(seed)
# np.random.seed(seed)

# class PredictWrapper(ClassifierMixin, object): # <-- ClassifierMixin 상속 추가
#     def __init__(self, estimator):
#         self.estimator = estimator
#         # 기본 Estimator의 _estimator_type을 전달 (필요한 경우)
#         if hasattr(estimator, '_estimator_type'):
#             self._estimator_type = estimator._estimator_type
#         else:
#             self._estimator_type = "classifier" # 분류기임을 명시

#     def fit(self, X, y, **kwargs):
#         self.estimator.fit(X, y, **kwargs)
#         return self

#     def predict(self, X):
#         pred = self.estimator.predict(X)
#         if pred.ndim > 1 and pred.shape[1] == 1:
#             return pred.ravel()
#         return pred

#     @property
#     def classes_(self):
#         return self.estimator.classes_

#     @property
#     def n_classes_(self):
#         if hasattr(self.estimator, 'n_classes_'):
#             return self.estimator.n_classes_
#         elif hasattr(self.estimator, 'classes_'):
#             return len(self.estimator.classes_)
#         return None 

# #1. 데이터
# datasets = load_digits()
# x = datasets.data
# y = datasets.target    

# le = LabelEncoder() # wine 데이터셋은 이미 정수 라벨이지만, 혹시 모를 상황에 대비하여 유지합니다.
# y = le.fit_transform(y)

# n_classes = len(le.classes_) 

# x_train, x_test, y_train, y_test = train_test_split(
#     x,y, shuffle=True, random_state=123, train_size=0.8, stratify=y
#     )

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델
# xgb_base = XGBClassifier(
#     objective='multi:softmax', # 다중 클래스 분류를 위한 objective
#     num_class=n_classes,     # 클래스 개수 명시
#     use_label_encoder=False, # DeprecationWarning 방지 및 최신 사용법 권장
#     eval_metric='mlogloss',  # 다중 분류 평가 지표
#     n_jobs=-1                # 모든 코어 사용
# )
# lg_base = LGBMClassifier(
#     objective='multiclass',  # 다중 클래스 분류를 위한 objective
#     num_class=n_classes,     # 클래스 개수 명시
#     n_jobs=-1                # 모든 코어 사용
# )
# cat_base = CatBoostClassifier(
#     loss_function='MultiClass', # 다중 클래스 분류를 위한 손실 함수
#     classes_count=n_classes,    # 클래스 개수 명시
#     verbose=0,                  # 훈련 중 출력되는 메시지 억제
# )

# xgb = PredictWrapper(xgb_base)
# lg = PredictWrapper(lg_base)
# cat = PredictWrapper(cat_base)

# model = VotingClassifier(
#     estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
#     # voting='soft',
#     voting='hard',   
# )

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# results = model.score(x_test, y_test)
# print('최종점수 : ', results)

# # 최종점수 :   # softvoting
# # 최종점수 :   # hardvoting

import numpy as np
import pandas as pd
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
import random
from bayes_opt import BayesianOptimization
from sklearn.datasets import load_digits 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score 
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
from sklearn.base import ClassifierMixin, BaseEstimator # <-- BaseEstimator 임포트

seed = 222
random.seed(seed)
np.random.seed(seed)

# **PredictWrapper 클래스를 BaseEstimator와 ClassifierMixin을 상속받도록 수정**
class PredictWrapper(ClassifierMixin, BaseEstimator): # <-- BaseEstimator 상속 추가
    def __init__(self, estimator):
        self.estimator = estimator
        # ClassifierMixin의 요구사항을 충족
        if hasattr(estimator, '_estimator_type'):
            self._estimator_type = estimator._estimator_type
        else:
            self._estimator_type = "classifier" 

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        pred = self.estimator.predict(X)
        if pred.ndim > 1 and pred.shape[1] == 1:
            return pred.ravel()
        return pred

    @property
    def classes_(self):
        return self.estimator.classes_

    @property
    def n_classes_(self):
        if hasattr(self.estimator, 'n_classes_'):
            return self.estimator.n_classes_
        elif hasattr(self.estimator, 'classes_'):
            return len(self.estimator.classes_)
        return None 

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target    

le = LabelEncoder() 
y = le.fit_transform(y)

n_classes = len(le.classes_) 
print(f"y의 고유 클래스 개수 (n_classes): {n_classes}") 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8,
    stratify=y 
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb_base = XGBClassifier(
    objective='multi:softmax', 
    num_class=n_classes,     
    use_label_encoder=False, 
    eval_metric='mlogloss',  
    n_jobs=-1                
)
lg_base = LGBMClassifier(
    objective='multiclass',  
    num_class=n_classes,     
    n_jobs=-1                
)
cat_base = CatBoostClassifier(
    loss_function='MultiClass', 
    classes_count=n_classes,    
    verbose=0,                  
)

xgb = PredictWrapper(xgb_base)
lg = PredictWrapper(lg_base)
cat = PredictWrapper(cat_base)

model = VotingClassifier(
    estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
    voting='hard',
    # voting='soft'   
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)


# 최종점수 :   # softvoting
# 최종점수 : 0.9805555555555555  # hardvoting



