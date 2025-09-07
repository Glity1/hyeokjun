# 27-5 copy

import sklearn as sk
print(sk.__version__)  
from sklearn.datasets import load_boston
import time
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

"""

from tensorflow.python.keras.layers import Dense, Dropout, LSTM

model.add(LSTM(10,activation='relu'))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)

test_csv = np.array(test_csv).reshape(test_csv.shape[0], test_csv.shape[1], 1)

"""
# 1.1 데이터 로드
datasets= load_boston()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names) 

x=datasets.data
y=datasets.target
print(x.shape, y.shape) #(506, 13) (506,)


x_train, x_test, y_train, y_test=train_test_split(
    x,y, train_size=0.8, shuffle=True,
    random_state=333,
)

scaler=RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape, x_test.shape) #(404, 13, 1) (102, 13, 1)
# exit()
# 2 모델 구성
model=Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(13, 1), padding='same', activation='relu'))         
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(11, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))

# 3 컴파일, 훈련 
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', 
    patience=50,
    restore_best_weights=True,  
)

hist = model.fit(x_train,y_train, epochs=100, batch_size=32, 
          verbose=3,                          
          validation_split=0.2,
          callbacks=[es],  # mcp 삽입
          )

#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict([x_test])
# print(x_test.shape) #(102, 13, 1)


print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

rmse= RMSE(y_test, results)
print("loss : ", loss)
print('RMSE:', rmse)
print('r2 스코어 : ', r2)
