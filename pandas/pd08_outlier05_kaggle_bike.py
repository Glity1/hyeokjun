#49_06 copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn. metrics import r2_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import time
import random
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

seed = 123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = ('./_data/kaggle/bike/') 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

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

pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)  #각각의 컬럼 데이터값끼리의 곱하기만 출력 제곱을 빼면 성능이 좀 떨어짐.
x_pf = pf.fit_transform(x_clean)

feature_names = [f"f{i}" for i in range(x_pf.shape[1])]

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y_clean, test_size=0.2, random_state=seed, #stratify=y,
)

#2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = 'rmse',
    data_name = 'validation_0',
    # save_best = True,
)

model = XGBRegressor(
    n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,
    eval_metric = 'rmse',  # 다중분류 : mlogloss, merror, 
                              # 이진분류 : logloss, error
                              # 회귀 : rmse, mae, mrsle
                              # 2.1.1 버전 이후로 fit에서 모델로 위치이동.
    callbacks = [es],
    random_state=seed)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 0,
          )    

print('r2 : ', model.score(x_test, y_test))   # acc2 :  0.9666666666666667

score_dict = model.get_booster().get_score(importance_type='gain')   # gain으로 중요도 기여도 판별
total = sum(score_dict.values())

# print("score_dict:", score_dict)
# print("feature_names[:5]:", feature_names[:5])
# exit()
score_list = [score_dict.get(col, 0) / total for col in feature_names]

thresholds = np.sort(score_list)  # sort 오름차순 정렬 낮은게 젤 앞으로 점점 커지는것
 
###### 컬럼명 매칭 #######  DataFrame 만드는 ㄴ연습했다
# score_df = pd.DataFrame({
#     # 'feature' : [feature_names[int(f[1:])]    for f in score_dict.keys()],
#     'feature' : feature_names,
#     'gain'    : list(score_dict.values())                                                   # gain의 수치는 수치만큼 오차를 줄였다 수치가 크면 클수록 좋다
#     'gain'    : score_list                                                   # gain의 수치는 수치만큼 오차를 줄였다 수치가 크면 클수록 좋다
# }).sort_values(by='gain', ascending=True)     # 오름차순 True 내림차순 False
# print(score_df)
# exit()
delete_columns = []
max_r2 = 0

for i in thresholds : 
    selection = SelectFromModel(model, threshold=i, prefit=False)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    if select_x_train.shape[1] == 0:
        print(f"Thresh={i:.3f},  No features selected → Skipping")
        continue
    
    # print(select_x_train.shape)

    select_model = XGBRegressor(
        n_estimators = 500,
        max_depth = 6,
        gamma = 0,
        min_child_weight = 0,
        subsample = 0.4,
        reg_alpha = 0,
        reg_lambda = 1,
        eval_metric = 'rmse',  # 다중분류 : mlogloss, merror, 이진분류 : logloss, error
                                # 2.1.1 버전 이후로 fit에서 모델로 위치이동.
        # early_stopping_rounds=30,
        # callbacks = [es],
        random_state=seed
        )
    
    select_model.fit(select_x_train, y_train,
          eval_set = [(select_x_test, y_test)],
          verbose = 0,
          )    
    
    # feature_list = selection.get_support()            # selection.get_support()은 SelectFromModel 객체에서 선택된 feature들을 불리언 배열 형태로 반환 즉 True 또는 False로 구성된 NumPy 배열
    # removed_features = feature_names[~feature_list]   #~mask는 불리언 반전 을 통해 True 를 False로 False 는 True로 해준다
    # 한줄로 ▼▼▼▼▼▼▼
    
    select_y_pred = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_pred)     
    # removed_features = feature_names[~selection.get_support()]
    # accuracy(정답, 모델이 예측한 값) : 두개를 비교해서 정답을 얼마나 맞췄는지 정확도
    print('Thresh=%.3f, n=%dscore, R2: %.4f%%' %(i, select_x_train.shape[1], score*100))
    # print('삭제할 컬럼 (%d개):'% len(removed_features), list(removed_features))   # len()은 길이를 구하는 함수 // #
    print("==============================================")
    
    # mask = selection.get_support()
    # # print('선택된 feature : ', mask)
    # not_select_mask = [feature_names[j] 
    #             for j, selected in enumerate(mask) 
    #             if not selected]    # for index, value // mask를 j(index) 와 selected(value)로 출력할거다.

    removed_features = [name for name, score in zip(feature_names, score_list) if score < i]


    # print('Thresh=%.3f, n=%dscore, ACC: %.4f%%' % (i, select_x_train.shape[1], score*100))
    # if removed_features:
    #     print(f'삭제할 컬럼 ({len(removed_features)}개):', removed_features)
    # else:
    #     print('삭제할 컬럼이 없습니다. (모든 feature가 선택됨)')
    # print("==============================================")
    
    if score >= max_r2:
        max_r2 = score
        delete_columns = removed_features.copy()
        
print(' 가장 높은 정확도: %.4f%%' % (max_r2 * 100))
print(' 이때 삭제할 컬럼들 (%d개):' % len(delete_columns), delete_columns)   


# Thresh=0.082, n=7score, R2: 7.5694%
# ==============================================
# Thresh=0.086, n=6score, R2: 7.5865%
# ==============================================
# Thresh=0.093, n=5score, R2: 23.2149%
# ==============================================
# Thresh=0.103, n=4score, R2: 20.5395%
# ==============================================
# Thresh=0.112, n=3score, R2: 23.0162%
# ==============================================
# Thresh=0.132, n=2score, R2: 19.8661%
# ==============================================
# Thresh=0.141, n=1score, R2: 16.4829%
# ==============================================
# Thresh=0.250,  No features selected → Skipping
#  가장 높은 정확도: 23.2149%
#  이때 삭제할 컬럼들 (2개): ['holiday', 'weather']

# pf 적용시
#  가장 높은 정확도: 21.0965%

# 가장 높은 정확도: 20.0733%





























