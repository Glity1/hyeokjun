import tensorflow as tf
import numpy as np
import sklearn as sk
import pandas as pd
import time
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 완전 비활성화

#1. 데이터

path = ('./_data/kaggle/bike/bike-sharing-demand/') # 상대경로
# path = ('.\_data\kaggle\bike\bike-sharing-demand/') # \n \a \b 등 예약어를 제외하면 가능
# path = ('.//_data//kaggle//bike//bike-sharing-demand/')
# path = ('.\\_data\\kaggle\\bike\\bike-sharing-demand/')
# path = 'C:/Study25/_data/kaggle/bike/bike-sharing-demand/' # 절대경로
# / 와 \ 를 섞어서 쓸 수도 있다

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

#region 데이터 정보
print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(submission_csv.shape) # (6493, 2)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']


print(test_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed']

print(submission_csv.columns)
# Index(['datetime', 'count']

print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())    # 결측치 없음

print(train_csv.describe())     # 데이터의 통계
#endregion

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)        # [10886 rows x 8 columns]
y = train_csv['count']
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=7572
)

x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(8, 1), padding='same', activation='relu'))         
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='linear'))    # activation='linear'는 default값이라 생략가능

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 256,
          verbose=1, validation_split=0.2)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print(r2)

# 0.282783401884948

# dropout
# 0.2111162027142952

# ##
# 0.265705223115413