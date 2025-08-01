import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(0, 17))
y = np.array(range(0, 17))

# [실습] 리스트의 슬라이싱으로 10:4:3 으로 나눈다.

x_train = x[:10]
y_train = y[:10]
x_val   = x[10:14]
y_val   = y[10:14]
x_test  = x[14:]
y_test  = y[14:]

print(x_train)          # [0 1  2  3  4  5  6  7  8  9 ]
print(y_train)          # [0 1  2  3  4  5  6  7  8  9 ]
print(x_val)            # [10 11 12 13]
print(y_val)            # [10 11 12 13]
print(x_test)           # [14 15 16]
print(y_test)           # [14 15 16]

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 ,훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=200, batch_size=1,
          verbose=1,
          validation_data=(x_val, y_val)                # 머신이 훈련하면서 검증까지 가중치 갱신에 직접적인 도움 X
          )

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([17])

print('loss : ', loss)
print('[17]의 예측값 : ', results)
