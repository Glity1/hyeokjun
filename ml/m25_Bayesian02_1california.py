from bayes_opt import BayesianOptimization
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.experimental import enable_halving_search_cv           # 정식버전이 아니라서 적어주는 라인
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
import time

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55,)

#2. 모델
def xgb_model(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, reg_alpha):
    model = XGBRegressor(
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2', n_jobs=-1)
    mean_score = score.mean()


beyesian_params = {
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3,10),
    'min_child_weight' : (1, 50),
}

optimizer = BayesianOptimization(
    f = xgb_model,                   # 함수는 y_function을 쓴다.
    pbounds=beyesian_params,
    random_state=333,
)

optimizer.maximize(init_points=5,     # 초기 훈련 5번     # 총 25번 돌려라
                   n_iter=20)         # 반복 훈련 20번

print(" 최적의 값 : ", optimizer.max)

