from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import time
import pandas as pd
import numpy as np
import random

seed = 123
np.random.seed(seed)
random.seed(seed)

#1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
feature_names = train_csv.columns

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome'] 

import matplotlib.pyplot as plt
def outlier(nums):
    q1, q2, q3 = np.percentile(nums, [25, 50, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    loc = np.where((nums > upper) | (nums < lower))
    return loc, iqr, upper, lower

for i in range(x.shape[1]):                                            # aaa.shape[1]의 의미는 2 즉 숫자, 총 열의 개수를 말한다 
    print(f"\n열 {i}에 대한 이상치 검사:")
    nums = x.iloc[:, i]
    outlier_loc, iqr, up, low = outlier(nums)
    print('이상치 위치:', outlier_loc)
    plt.figure()
    plt.boxplot(x.iloc[:, i])
    plt.axhline(up, color='red', linestyle='--', label='upper bound')
    plt.axhline(low, color='blue', linestyle='--', label='lower bound')
    plt.title(f'열 {i} boxplot')
    plt.legend()
    # plt.show()

def remove_outliers(x, y):
    total_outlier_idx = set()
    for i in range(x.shape[1]):
        nums = x.iloc[:, i]
        outlier_loc, _, _, _ = outlier(nums)
        total_outlier_idx.update(outlier_loc[0])
    print(f'제거할 이상치 행 수: {len(total_outlier_idx)}')
    mask = np.ones(len(x), dtype=bool)
    mask[list(total_outlier_idx)] = False
    return x[mask], y[mask]

x_clean, y_clean = remove_outliers(x, y)
    
pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)  #각각의 컬럼 데이터값끼리의 곱하기만 출력 제곱을 빼면 성능이 좀 떨어짐.
x_pf = pf.fit_transform(x_clean)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y_clean, stratify=y_clean,test_size=0.2, random_state=55
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) #(521, 44)
# exit()
# 4. PCA 적용
pca = PCA(n_components=32)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# 5. XGBoost 모델 학습
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    gamma=0,
    min_child_weight=0,
    subsample=0.4,
    reg_alpha=0,
    reg_lambda=1,
    eval_metric='logloss',
    random_state=seed
)

model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0)
print('Base acc Score:', accuracy_score(y_test, model.predict(x_test)))

# 6. 중요도 기반 정렬
score_dict = model.get_booster().get_score(importance_type='gain')
total_gain = sum(score_dict.values())
score_list = [score_dict.get(f"f{i}", 0) / total_gain for i in range(x_train.shape[1])]
thresholds = np.sort(score_list)

# 7. 반복적으로 특성 제거 후 성능 확인
max_score = 0
delete_indices = []

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    if select_x_train.shape[1] == 0:
        print(f"Thresh={thresh:.6f}, No features selected → Skipping")
        continue

    temp_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        gamma=0,
        min_child_weight=0,
        subsample=0.4,
        reg_alpha=0,
        reg_lambda=1,
        eval_metric='logloss',
        random_state=seed
    )

    temp_model.fit(select_x_train, y_train, eval_set=[(select_x_test, y_test)], verbose=0)
    y_pred = temp_model.predict(select_x_test)
    score = accuracy_score(y_test, y_pred)

    removed = [i for i, s in enumerate(score_list) if s < thresh]
    print(f"Thresh={thresh:.6f}, n={select_x_train.shape[1]}, acc: {score:.4f}")

    if score >= max_score:
        max_score = score
        delete_indices = removed.copy()

# 8. 결과 출력
print('\n 가장 높은 (acc): %.4f' % max_score)
print(' 삭제할 주성분 인덱스 (%d개):' % len(delete_indices), delete_indices)


#가장 높은 (acc): 0.7557
#  삭제할 주성분 인덱스 (8개): [1, 2, 12, 18, 21, 22, 25, 29]

# 결측치삭제
# 가장 높은 (acc): 0.7570
#  삭제할 주성분 인덱스 (18개): [0, 1, 2, 3, 5, 7, 10, 12, 13, 15, 17, 18, 19, 20, 22, 24, 27, 30]