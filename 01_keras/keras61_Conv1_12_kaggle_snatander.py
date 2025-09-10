# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview
import numpy as np
import sklearn as sk
import pandas as pd
import time
import datetime
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

#1. 데이터
path = './_data/kaggle/santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# print(train_csv.shape)  # (200000, 201)
# print(test_csv.shape)   # (200000, 200)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55, stratify=y
)

scaler = RobustScaler()
x = scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(160000, 200) (40000, 200)
# exit()

x_train = x_train.reshape(x_train.shape[0], 200, 1)
x_test = x_test.reshape(x_test.shape[0], 200, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(200, 1), padding='same', activation='relu'))         
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(
    x_train, y_train, epochs=1, batch_size=512,
    callbacks=[es], validation_split=0.2, verbose=2
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results)
y_predict = model.predict(x_test)
roc = roc_auc_score(y_test, y_predict)
print('roc : ', roc)
print('runtime : ', end_time - start_time)

# loss :  [0.2433519959449768, 0.9100499749183655]
# roc :  0.8459235886990839
# runtime :  30.39808940887451

# dropout
# loss :  [0.24047136306762695, 0.9113249778747559]
# roc :  0.8485828708043993
# runtime :  36.03268766403198