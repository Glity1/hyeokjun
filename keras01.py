import tensorflow as tf
print(tf.__version__)   # 2.9.3  
import numpy as np
print(np.__version__)   # 1.21.1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6]) # x는 1,2,3 ~ 이다  y=x라는 수식이 들어가있음
y = np.array([1,2,3,4,5,6]) # y는 1,2,3 ~ 이다

#2. 모델구성
model = Sequential() #Sequential 이라는 모델을 만들꺼다
model.add(Dense(1, input_dim=1)) #Dense 모델을 들어가는게 1개 나가는게 1개인 모델을 만들꺼다

#3. 컴파일(컴퓨터가 알아먹게 하는것), 훈련
model.compile(loss='mse', optimizer='adam') #loss: 원데이터와 예상 데이터의 오차
model.fit(x, y, epochs=3500) # fit 훈련시키겠다 x와y를 훈련시킬껀데 epochs 반복 3500하겠다 5가 5라고 답할때까지 epochs를 늘린다 / 적절한 값을 찾아야함
# 최적의 가중치를 찾아가는 과정
 
#4. 평가, 예측
result= model.predict(np.array([7])) # 7의 값을 예측
print('7의 예측값 : ', result) # 값을 출력