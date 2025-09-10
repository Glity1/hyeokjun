#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import sklearn as sk
print(sk.__version__) # 1.1.3
import tensorflow as tf
print(tf.__version__) # 2.9.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names) #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

x = datasets.data
y = datasets.target #통상적으로 target 은 y값

print(x)
print(y)
print(x.shape, y.shape) # x 형태 (20640, 8) y 형태 (20640,) 행은 항상 같다
                        # 소수점으로 되있으면 회귀모델
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813
)

#2. 모델구성
model = Sequential()
model.add(Dense(80, input_dim=8))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train,y_train, epochs=3500, batch_size=64) 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

print("loss : ", loss)
print("[x_test]의 예측값 : ", results)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)

# 목표 r2 0.59이상

# loss :   0.5334346294403076

# r2 스코어 :   0.6027431852313392
#     test_size=0.2,
#     random_state=813

# model.add(Dense(80, input_dim=8))
# model.add(Dense(80))
# model.add(Dense(60))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(12))
# model.add(Dense(1))

# model.fit(x_train,y_train, epochs=3500, batch_size=64) #epochs,layer를 늘려서 목표달성