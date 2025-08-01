#컬럼, 열 갯수에 따른 성능테스트

import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

#1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])
y = np.array([1,2,3,4,5])
#x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
x = np.transpose(x)

print(x.shape) # (5, 2) 데이터구조를 받고나서 뒤에 숫자 2가 컬럼 input_dim=2가 된다.
print(y.shape) # (5,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2)) # dim 컬럼의 갯수를 명시
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[6, 11]]) # (1, 2)
print('loss :', loss)
print('[6,11]의 예측값 ', results)

#결과
# loss : 6.963318810448982e-13
# [6,11]의 예측값  [[6.]]