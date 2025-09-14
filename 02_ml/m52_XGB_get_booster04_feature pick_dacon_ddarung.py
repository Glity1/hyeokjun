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

seed = 123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)   
y = train_csv['count']
feature_names = train_csv.columns
feature_names = list(x.columns)
# print(feature_names)
# ['hour', 'hour_bef_temperature', 'hour_bef_precipitation', 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility', 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count']
# exit()


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed,
)
print(x_train.shape) #(1167, 9)

#2. 모델


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
    random_state=seed)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 0,
          )    
print(x_train.shape)

print('r2 : ', model.score(x_test, y_test))   # acc2 :  0.9666666666666667

score_dict = model.get_booster().get_score(importance_type='gain')   # gain으로 중요도 기여도 판별
# total = sum(score_dict.values())
print(score_dict) #{'hour': 929.53857421875, 'hour_bef_temperature': 715.8067626953125, 'hour_bef_precipitation': 2706.689697265625, 'hour_bef_windspeed': 218.9258575439453, 'hour_bef_humidity': 225.1244659423828, 'hour_bef_visibility': 296.1829833984375, 'hour_bef_ozone': 295.91851806640625, 'hour_bef_pm10': 238.1697998046875, 'hour_bef_pm2.5': 245.9815673828125}
# exit()
# score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])] 
score_list =  np.array(list(score_dict.values()))

score_list_sc = score_list / np.sum(score_list)
print(score_list)

# [ 929.53857422  715.8067627  2706.68969727  218.92585754  225.12446594
#   296.1829834   295.91851807  238.1697998   245.98156738]
# exit()
thresholds = np.sort(score_list_sc)  # sort 오름차순 정렬 낮은게 젤 앞으로 점점 커지는것
print(thresholds)
#[ 218.92585754  225.12446594  238.1697998   245.98156738  295.91851807
#   296.1829834   715.8067627   929.53857422 2706.68969727]
# exit()
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
    print(select_x_train.shape)
    
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















