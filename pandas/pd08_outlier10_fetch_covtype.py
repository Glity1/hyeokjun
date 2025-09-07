from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import time
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
x, y = fetch_covtype(return_X_y=True)
y = y -1

def outlier(nums):
    q1, q2, q3 = np.percentile(nums, [25, 50, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    loc = np.where((nums > upper) | (nums < lower))
    return loc, iqr, upper, lower

for i in range(x.shape[1]):                                            # aaa.shape[1]의 의미는 2 즉 숫자, 총 열의 개수를 말한다 
    print(f"\n열 {i}에 대한 이상치 검사:")
    nums = x[:, i]
    outlier_loc, iqr, up, low = outlier(nums)
    print('이상치 위치:', outlier_loc)
    plt.figure()
    plt.boxplot(x[:, i])
    plt.axhline(up, color='red', linestyle='--', label='upper bound')
    plt.axhline(low, color='blue', linestyle='--', label='lower bound')
    plt.title(f'열 {i} boxplot')
    plt.legend()
    # plt.show()

def replace_outliers_with_median(x):
    x_copy = x.copy()
    for i in range(x.shape[1]):
        nums = x[:, i]
        outlier_loc, _, upper, lower = outlier(nums)
        median = np.median(nums)
        x_copy[outlier_loc, i] = median
    return x_copy

# 4. 이상치 대체 수행
x_clean = replace_outliers_with_median(x)
y_clean = y

print(f'x_clean shape: {x_clean.shape}')
print(f'y_clean shape: {y_clean.shape}')
# exit()

pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)  #각각의 컬럼 데이터값끼리의 곱하기만 출력 제곱을 빼면 성능이 좀 떨어짐.
x_pf = pf.fit_transform(x_clean)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y_clean, test_size=0.2, stratify=y_clean, random_state=55
)

n_split = 3
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=503)

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12], 'learning_rate': [0.1, 0.01, 0.001]},    # 18
    {'max_depth': [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]},                            # 12
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}                       # 12
]

#2. 모델
xgb = XGBClassifier()
model = GridSearchCV(xgb, parameters, cv=kfold,   # 42 * 5 = 210
                     verbose=2,
                     n_jobs=1,
                     refit=True,    # 1번
                     )  # 총 210번 + 1번 = 211번

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start

print('\n\n- 최적의 매개변수 : ', model.best_estimator_)
print('- 최적의 파라미터 : ', model.best_params_)

#4. 평가, 예측
print('- best_score : ', model.best_score_)
print('- mode.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('- accuracy_score : ', accuracy_score(y_test, y_pred))
print('- 걸린시간 : ', round(end, 2), '초\n\n')

# - 최적의 매개변수 :  XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=12,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=500,
#               n_jobs=None, num_parallel_tree=None, ...)
# - 최적의 파라미터 :  {'learning_rate': 0.1, 'max_depth': 12, 'n_estimators': 500}
# - best_score :  0.9678427043661786
# - mode.score :  0.970904365635999
# - accuracy_score :  0.970904365635999
# - 걸린시간 :  1863.9 초