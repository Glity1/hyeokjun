# https://dacon.io/competitions/open/235576/overview/description

import numpy as np                                           # 훈련에 특화됨.
import pandas as pd                                          # 데이터분석을 할 때 전처리 정제에서 유명함.
print(np.__version__)                                        # 1.23.0
print(pd.__version__)                                        # 2.2.3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/dacon/따릉이/' # .(점 한개) = 현재 작업폴더 study25

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # a=b b를 a에 넣겠다. // index_col : 이 컬럼은 인덱스다
print(train_csv) # [1459 rows x 10 columns] = [1459,10]                  

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) # [715 rows x 9 columns] = [715,9]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv) # [715 rows x 1 columns] = [715,9]  // Nan 데이터가 없음 : 결측치

print(train_csv.shape)              #(1459,10)
print(test_csv.shape)               #(715,9)
print(submission_csv.shape)         #(715,1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_csv.info())

print(train_csv.describe())

x = 
x_train, x_test, y_train, y_test = train_test_split(
    train_csv,test_csv,
    test_size=0.1,
    random_state=813
)


#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=10))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs =100, batch_size=10)

#4. 평가, 예측
loss = model.evaluate(train_csv, submission_csv)
results = model.predict([x_test])

print('loss : ', loss)
print("[x_test]의 예측값 : ", results)

r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)


