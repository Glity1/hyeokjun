# 27-5 copy

import sklearn as sk
print(sk.__version__)  
from sklearn.datasets import load_boston
import time
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1.1 데이터 로드
datasets= load_boston()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names) 

x=datasets.data
y=datasets.target
print(x.shape, y.shape) 

x_train, x_test, y_train, y_test=train_test_split(
    x,y, train_size=0.8, shuffle=True,
    random_state=333,
)

print(x_train.shape, x_test.shape)  #(404, 13) (102, 13)
print(y_train.shape, y_test.shape)  #(404,) (102,)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

# exit()

x_train = x_train.reshape(404,13,1,1)
x_test = x_test.reshape(102,13,1,1)
  
# 2 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, padding='same', input_shape=(13, 1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Activation('relu')) 
model.add(Dropout(0.1))
model.add(Dense(units=1, activation='linear'))

# 3 컴파일, 훈련 (19_1  copy)
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

print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

rmse= RMSE(y_test, results)
print("loss : ", loss)
print('RMSE:', rmse)
print('r2 스코어 : ', r2)

