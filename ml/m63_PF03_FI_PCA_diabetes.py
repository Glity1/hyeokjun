# Final version: PCA + Feature Importance + SelectFromModel
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import warnings
import random
warnings.filterwarnings('ignore')

# 설정
seed = 123
np.random.seed(seed)
random.seed(seed)

# 1. 데이터 로드
datasets = load_diabetes()
x = datasets.data
y = datasets.target
feature_names = datasets.feature_names

# 2. PolynomialFeatures 적용
pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
x_pf = pf.fit_transform(x)

# 3. Train/Test 분할 + 스케일링
x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, test_size=0.2, random_state=seed)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. PCA 적용
pca = PCA(n_components=10)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# 5. XGBoost 모델 학습
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    gamma=0,
    min_child_weight=0,
    subsample=0.4,
    reg_alpha=0,
    reg_lambda=1,
    eval_metric='rmse',
    random_state=seed
)

model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0)
print('Base R² Score:', r2_score(y_test, model.predict(x_test)))

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

    temp_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        gamma=0,
        min_child_weight=0,
        subsample=0.4,
        reg_alpha=0,
        reg_lambda=1,
        eval_metric='rmse',
        random_state=seed
    )

    temp_model.fit(select_x_train, y_train, eval_set=[(select_x_test, y_test)], verbose=0)
    y_pred = temp_model.predict(select_x_test)
    score = r2_score(y_test, y_pred)

    removed = [i for i, s in enumerate(score_list) if s < thresh]
    print(f"Thresh={thresh:.6f}, n={select_x_train.shape[1]}, R²: {score:.4f}")

    if score >= max_score:
        max_score = score
        delete_indices = removed.copy()

# 8. 결과 출력
print('\n 가장 높은 정확도 (R²): %.4f' % max_score)
print(' 삭제할 주성분 인덱스 (%d개):' % len(delete_indices), delete_indices)

#  가장 높은 정확도 (R²): 0.4093
#  삭제할 주성분 인덱스 (5개):  [0, 1, 7, 8, 9]