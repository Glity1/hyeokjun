import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())
y = train_csv['Outcome'] 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)                   # StratifiedKFold 범주형일 때 확실하게 분류 // 데이터가 많으면 kfold써도됨.

#2. 모델구성
model = MLPClassifier()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc : ", scores, "\n 평균 acc : ", round(np.mean(scores), 4))

# acc :  [0.74285714 0.77884615 0.82692308 0.77884615 0.80769231] 
#  평균 acc :  0.787

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)                                  # test 로 진행
# y_pred = np.round(y_pred)
                 
acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc) # cross_val_predict ACC :   0.6641221374045801


