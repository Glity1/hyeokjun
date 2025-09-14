#kaggle_bank

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

le = LabelEncoder()
for col in ['Geography', 'Gender']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

train_csv = train_csv.drop(["CustomerId","Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId","Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited'] 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333) 

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')

max_acc = 0
max_name = '최고'

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        #3. 훈련
        model.fit(x_train, y_train)
        
        #4. 평가, 예측
        results = model.score(x_test, y_test)
        print(name, '의 정답률 : ', results)    
       
        if results > max_acc:
           max_acc = results
           max_name = name        
        
    except:
        print(name, "은(는) 정답률을 생성할 수 없습니다!")
        
print("==============================================================================================")
print("최고모델 : ", max_name, max_acc)
print("==============================================================================================")

# 최고 모델 클래스 객체 가져오기
best_model_class = None
for name, algorithms in allAlgorithms:
    if name == max_name:
        best_model_class = algorithms
        break

# 예외 처리: 못 찾은 경우
if best_model_class is None:
    raise ValueError(f"{max_name} 모델 클래스를 찾을 수 없습니다.")

# 최고 모델 인스턴스화
best_model = best_model_class()

#3. 훈련
scores = cross_val_score(best_model, x_train, y_train, cv=kfold)
print("acc : ", scores, "\n 평균 acc : ", round(np.mean(scores), 4))

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)                                  # test 로 진행
                 
acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc)

