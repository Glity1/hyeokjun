import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]
              ])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape)                    # (13, 3) (13,)
x = x.reshape(x.shape[0], x.shape[1], 1)  
print(x.shape)                             # (13, 3, 1)

# exit()

#2. 모델구성
model = Sequential()
model.add(LSTM(16, input_shape=(3, 1), return_sequences=True,activation='relu'))         #3차원 데이터로 들어가고 2차원 데이터로 결과가 나온다.
model.add(GRU(16, return_sequences=True, activation='relu'))
model.add(SimpleRNN(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 3, 16)             1152

#  dense (Dense)               (None, 3, 16)             272

#  dense_1 (Dense)             (None, 3, 8)              136

#  dense_2 (Dense)             (None, 3, 1)              9

# =================================================================
# Total params: 1,569
# Trainable params: 1,569
# Non-trainable params: 0
# _________________________________________________________________

# exit()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2500,) #callbacks = [mcp])

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)

x_pred = np.array([50,60,70]).reshape(1,3,1)      # (3,) -> (1,3,1) reshape
y_pred = model.predict(x_pred)


print('[50,60,70]의 결과 : ', y_pred)


# loss :  2.9033869492628428e-12
# [50,60,70]의 결과 :  [[80.00002]]