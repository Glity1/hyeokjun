from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np 

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성 (딥러닝을 모델로 구현했을 때)
model = Sequential()
model.add(Dense(10, input_dim=1)) #처음 인풋은 수정불가
model.add(Dense(100, input_dim=10)) # 각 수치하나를  파라미터라고함
model.add(Dense(100, input_dim=100))
model.add(Dense(1, input_dim=100)) #마지막 아웃풋 수정불가, 그림참조
 
epochs = 300 # epochs 이름변경 가능
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs) # 우측 epochs 이름변경 16라인과 동일하다면 가능

#4. 평가, 예측
loss = model. evaluate(x,y) 
print('#################################')
print('epochs : ', epochs)
print('loss : ', loss)
results = model.predict([6])
print('6의 예측값 : ', results)

#################################
# epochs :  300
# loss :  3.0002578910170996e-07
# 1/1 [==============================] - 0s 63ms/step
# 6의 예측값 :  [[5.999173]]