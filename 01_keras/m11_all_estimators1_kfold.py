# california
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)  

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='regressor')

max_score = 0
max_name = '최고'

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        #3. 훈련
        model.fit(x_train, y_train)
        
        #4. 평가, 예측
        results = model.score(x_test, y_test)
        print(name, '의 정답률 : ', results)    
       
        if results > max_score:
           max_score = results
           max_name = name        
        
    except:
        print(name, "은(는) 정답률을 생성할 수 없습니다!")    # 버전문제나 파라미터가 입력안되서 문제가 생겼음.
        
print("==============================================================================================")
print("최고모델 : ", max_name, max_score)
print("==============================================================================================")

best_model_class = None
for name, algorithms in allAlgorithms:
    if name == max_name:
        best_model_class = algorithms
        break

if best_model_class is None:
    raise ValueError(f"{max_name} 모델 클래스를 찾을 수 없습니다.")

best_model = best_model_class()

#3. 훈련
scores = cross_val_score(best_model, x_train, y_train, cv=kfold)
print("r2_score : ", scores, "\n 평균 r2_score : ", round(np.mean(scores), 4))

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)                                  
                 
scores = r2_score(y_test, y_pred)
print('cross_val_predict scores : ', scores)

# ==============================================================================================
# 최고모델 :  HistGradientBoostingRegressor 0.8273646768010888
# ==============================================================================================
# r2_score :  [0.83321416 0.83509314 0.83847358 0.83234954 0.82993368] 
#  평균 r2_score :  0.8338
# cross_val_predict scores :  0.019541401701086314