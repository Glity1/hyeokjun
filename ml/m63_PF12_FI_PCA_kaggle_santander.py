from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import time
import pandas as pd
import numpy as np
import random

seed = 123
np.random.seed(seed)
random.seed(seed)

#1. 데이터
path = './_data/kaggle/santander/'                                          
train_csv = pd.read_csv(path + 'train.csv', index_col=0)                    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)                      

x = train_csv.drop(['target'], axis=1)                                     
y = train_csv['target']                                                    

feature_names = list(x.columns)

pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)  #각각의 컬럼 데이터값끼리의 곱하기만 출력 제곱을 빼면 성능이 좀 떨어짐.
x_pf = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, stratify=y,test_size=0.2, random_state=55
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) #(132027, 65)
# exit()
# 4. PCA 적용
pca = PCA(n_components=64)
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


# 가장 높은 (acc): 0.9737