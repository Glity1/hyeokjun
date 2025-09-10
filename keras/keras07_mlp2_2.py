# range 를 활용한 딥러닝 계산
import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) #10직전까지 개수를 산정한다. -1하면 편하다
print(x) # [0 1 2 3 4 5 6 7 8 9] 처음시작을 0부터 시작한다. 
print(x.shape) # (10,)

x = np.array(range(1, 10))
print(x) # [1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 11))
print(x) # [ 1  2  3  4  5  6  7  8  9 10]

x = np.array([range(10), range(21, 31), range(201, 211)]) #
print(x)
print(x.shape) #(3, 10)

x = x.T # .transpose
print(x)
print(x.shape) # (10, 3)

y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습]
#[10, 31, 211] 예측

#2. 모델구성
model = Sequential()
model.add(Dense (10, input_dim=3))
model.add(Dense (10))
model.add(Dense (10))
model.add(Dense (10))
model.add(Dense (10))
model.add(Dense (10))
model.add(Dense (1))


#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x,y, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print('loss : ', loss)
print('[10, 31, 221]의 예측값 ', results)

#5. 결과
# loss : 1.2846612879383046e-12
# [11, 2.0, -1]의 예측값  [[11.000001]]
