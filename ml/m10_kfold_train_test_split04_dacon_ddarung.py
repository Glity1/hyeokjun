import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

train_csv = train_csv.fillna(train_csv.mean())    
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1) 
y = train_csv['count']  

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)                   # StratifiedKFold 범주형일 때 확실하게 분류 // 데이터가 많으면 kfold써도됨.

#2. 모델구성
model = HistGradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc : ", scores, "\n 평균 acc : ", round(np.mean(scores), 4))

# acc :  [0.76102636 0.75264813 0.78878259 0.82815858 0.76858036] 
#  평균 acc :  0.7798

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)                                  # test 로 진행
y_pred = np.round(y_pred)
                 
acc = r2_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc) # cross_val_predict ACC :  0.7047889577817423 



