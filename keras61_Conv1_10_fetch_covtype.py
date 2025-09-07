# import ssl
# import certifi

# SSL 인증서 문제 해결
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import numpy as np
import pandas as pd

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target


y = y.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)
print(y)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50, #stratify=y
)
print(x_train.shape, x_test.shape) #(464809, 54) (116203, 54)
# exit()

x_train = x_train.reshape(x_train.shape[0], 54, 1)
x_test = x_test.reshape(x_test.shape[0], 54, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(54, 1), padding='same', activation='relu'))         
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss', mode='min', patience=10, restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1, batch_size=1000,
          validation_split=0.2, callbacks=es, verbose=2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_pred = model.predict(x_test)
y_round = np.round(y_pred)
f1 = f1_score(y_test, y_round, average='macro')
print('f1 : ', f1)

y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_true, y_pred)
print(round(acc_score, 4))

# stratify 비활성화
# loss :  0.16281147301197052
# acc :  0.9369896054267883
# f1 :  0.8968958725575326

# stratify 활성화
# loss :  0.16585153341293335
# acc :  0.7434684038162231
# f1 :  0.4560115272432264

# loss :  0.9700334072113037
# acc :  0.5136786699295044
