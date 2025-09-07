# 얘가 중요
# 27-3 copy

import sklearn as sk
print(sk.__version__)  
from sklearn.datasets import load_boston
import time
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))  

# 2 모델 구성
model=Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(1))

model.summary() 

path='./_save/keras26/'
# model.save(path+'keras26_1_save.h5')
model.save_weights(path+'keras26_5_save1.h5')

# 3 컴파일, 훈련 (19_1  copy)
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', 
    patience=50,
    restore_best_weights=True,  
)

################## mcp 세이브 파일명 만들기 ##########################
import datetime
date=datetime.datetime.now()
print(date)       # 2025-06-02 13:01:50.955316
print(type(date)) # <class 'datetime.datetime'>
date=date.strftime("%m%d_%H%M")
print(date)       # 0602_1306
print(type(date)) # <class 'str'>

path='./_save/keras27_mcp2/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5' 
# {epoch:04d}	 현재 에포크 번호를 4자리로 표시 (예: 1 → 0001, 15 → 0015)
# {val_loss:.4f} 검증 손실(val_loss)을 소수점 아래 4자리까지 표시 (예: 0.1234)
filepath=''.join([path, 'k27_', date, '_', filename])

print(filepath)
# ./_save/keras27_mcp2/k27_0602_1442_{epoch:04d}-{val_loss:.4f}.hdf5
# exit()
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    #가장 좋은 값 저장
    filepath=filepath   # 과적합이 되기 전 가장 일반화 성능이 좋았던 모델을 저장하기 위함.
)

hist = model.fit(x_train,y_train, epochs=100, batch_size=32, 
          verbose=3,                          
          validation_split=0.2,
          callbacks=[es, mcp],  # mcp 삽입
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


# loss :  25.80413055419922
# RMSE: 5.079776500794532
# r2 스코어 :  0.7369045663584933

