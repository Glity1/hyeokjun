import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_iris()      
x = datasets.data
y = datasets['target']


x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8, stratify=y
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)                   # StratifiedKFold 범주형일 때 확실하게 분류 // 데이터가 많으면 kfold써도됨.

#2. 모델구성
model = MLPClassifier()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc : ", scores, "\n 평균 acc : ", round(np.mean(scores), 4))

# acc :  [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333]                            # train 만 한거
# 평균 acc :  0.9583

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)                                  # test 로 진행
print(y_test) #[1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 2 2 1 2 2 1 1 1 1]                 
print(y_pred) #[1 0 2 2 0 0 1 1 2 0 0 2 1 1 1 1 0 0 0 0 0 2 2 2 1 1 1 1 1 1]                 # round처리 자동으로 해줌

acc = accuracy_score(y_test, y_pred)
print('corss_val_predict ACC : ', acc) # corss_val_predict ACC :  0.9333333333333333  



