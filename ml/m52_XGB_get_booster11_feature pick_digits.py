#49_06 copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits 
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

seed = 123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target    
feature_names = datasets.feature_names

le = LabelEncoder() 
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, #stratify=y,
)

#2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    # metric_name = 'mlogloss',
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
    eval_metric = 'mlogloss',  # 다중분류 : mlogloss, merror, 
                              # 이진분류 : logloss, error
                              # 회귀 : rmse, mae, mrsle
                              # 2.1.1 버전 이후로 fit에서 모델로 위치이동.
    callbacks = [es],
    random_state=seed)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 0,
          )    

print('acc : ', model.score(x_test, y_test))   # acc2 :  0.9666666666666667

score_dict = model.get_booster().get_score(importance_type='gain')   # gain으로 중요도 기여도 판별
total = sum(score_dict.values())

score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])] 

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
max_acc = 0

for i in thresholds : 
    selection = SelectFromModel(model, threshold=i, prefit=False)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    if select_x_train.shape[1] == 0:
        print(f"Thresh={i:.3f},  No features selected → Skipping")
        continue
    
    # print(select_x_train.shape)

    select_model = XGBClassifier(
        n_estimators = 500,
        max_depth = 6,
        gamma = 0,
        min_child_weight = 0,
        subsample = 0.4,
        reg_alpha = 0,
        reg_lambda = 1,
        eval_metric = 'mlogloss',  # 다중분류 : mlogloss, merror, 이진분류 : logloss, error
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
    score = accuracy_score(y_test, select_y_pred)     
    # removed_features = feature_names[~selection.get_support()]
    # accuracy(정답, 모델이 예측한 값) : 두개를 비교해서 정답을 얼마나 맞췄는지 정확도
    print('Thresh=%.3f, n=%dscore, ACC: %.4f%%' %(i, select_x_train.shape[1], score*100))
    # print('삭제할 컬럼 (%d개):'% len(removed_features), list(removed_features))   # len()은 길이를 구하는 함수 // #
    print("==============================================")
    
    # mask = selection.get_support()
    # # print('선택된 feature : ', mask)
    # not_select_mask = [feature_names[j] 
    #             for j, selected in enumerate(mask) 
    #             if not selected]    # for index, value // mask를 j(index) 와 selected(value)로 출력할거다.

    removed_features = [feature_names[j] for j in range(len(score_list)) if score_list[j] < i]

    # print('Thresh=%.3f, n=%dscore, ACC: %.4f%%' % (i, select_x_train.shape[1], score*100))
    # if removed_features:
    #     print(f'삭제할 컬럼 ({len(removed_features)}개):', removed_features)
    # else:
    #     print('삭제할 컬럼이 없습니다. (모든 feature가 선택됨)')
    # print("==============================================")
    
    if score >= max_acc:
        max_acc = score
        delete_columns = removed_features.copy()
        
print(' 가장 높은 정확도: %.4f%%' % (max_acc * 100))
print(' 이때 삭제할 컬럼들 (%d개):' % len(delete_columns), delete_columns)   














