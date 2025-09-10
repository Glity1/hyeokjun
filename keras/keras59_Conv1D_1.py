#52-1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Flatten

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array(([1,2,3],                    # 7행 3열  time setp 3 // 3시간 단위로 자름 
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ))

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)                  #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)                           #(7, 3, 1)  // (batch_size, time setps, feature) : batch_size만큼 훈련시키겠다

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, 
                 padding= 'same', input_shape=(3,1)))   # (N, 3, 10)
model.add(Conv1D(10,2))                                  # (N, 2, 10)
model.add(Flatten())                                    # (N, 20)
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()
exit()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2400)

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)

x_pred = np.array([8,9,10]).reshape(1,3,1)      # (3,) -> (1,3,1) reshape
y_pred = model.predict(x_pred)


print('[8,9,10]의 결과 : ', y_pred)




