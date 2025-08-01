from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]])
x = x.T
y=  y.T
print(x)
print(x.shape) #(3, 10) -> (10, 3)
print(y)
print(y.shape) #(2, 10) -> (10, 2)

# [실습]
# loss와 [[10,31,221]]을 예측하시오 컬럼을 맞춰주기위한 대괄호가 두개인 이유 데이터구조를 맞춰준것

#2. 모델구성
model = Sequential()
model.add(Dense (10, input_dim=3))
model.add(Dense (20))
model.add(Dense (20))
model.add(Dense (20))
model.add(Dense (20))
model.add(Dense (20))
model.add(Dense (2))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x,y, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print('loss : ', loss)
print('[[10, 31, 221]]의 예측값 ', results)

# loss :  8.538467227481306e-05
# [[10, 31, 221]]의 예측값  [[10.997046    0.01152682]]