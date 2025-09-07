import numpy as np
import pandas as pd
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# print(x.shape)  # (1797, 64)
# print(y.shape)  # (1797,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
)

scaler = StandardScaler()
x = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 64, 1)
x_test = x_test.reshape(x_test.shape[0], 64, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(64, 1), padding='same', activation='relu'))         
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor='val_loss',         # 모니터 할 값
    mode='min',                 # 최대값 max, 알아서 찾기 auto
    patience=40,                # 40번 안에 최소값이 안나오면 멈춤
    restore_best_weights=True,  # 최소값을 저장할거면 True, default는 False
)

hist = model.fit(x_train, y_train, epochs = 1, batch_size = 128,
          verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])  # loss
print('acc : ', loss[1])  # acc
results = model.predict(x_test)

# loss :  1.2977604866027832

# dropout
# loss :  1.0834016799926758

# loss :  9.394128799438477
# acc :  0.10277777910232544