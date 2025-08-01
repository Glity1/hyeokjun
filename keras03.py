from tensorflow.keras.models import Sequential #노란밑줄 인식불가능
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model. add(Dense(1, input_dim=1))

epochs = 3500

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs) #100번째 가중치가 들어가있음 100번중에 제일 좋은 가중치는 아님

print("################################################################")

#4. 평가, 예측
loss= model. evaluate(x,y) #최종훈련시킨 가중치의 값의 loss 값을 볼 수 있음
print('epochs : ', epochs)
print('로스 : ', loss)
results = model.predict([7])
print("예측값 : ", results)
