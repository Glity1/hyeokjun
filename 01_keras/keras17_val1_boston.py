# 가독성 있게 배치할것
import sklearn as sk
print(sk.__version__) #1.6.1 -> 1.1.3

from sklearn.datasets import load_boston  # l을 치면 교육용 데이터셋들이 있음
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets = load_boston()
print(datasets)
print(datasets.DESCR) #(506,13) 데이터 DESCR : 묘사 (describe)
print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']

x = datasets.data
y = datasets.target

print(x)
print(x.shape) #(506, 13)
print(y)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=814   # validation_split로도 바뀌지않는다면 바꾸자
)
#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs=500, batch_size=1,
          verbose=1,
          validation_split=0.2)

#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict([x_test])

print("loss : ", loss)
print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)
# r2 기준으로 0.75이상
# loss :   29.726648330688477
# r2 스코어 : 0.7584895173069439
# r2 스코어 :  0.8484866157781902 로 상승