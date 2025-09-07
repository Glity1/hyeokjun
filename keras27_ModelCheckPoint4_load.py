# 27-2 copy

import sklearn as sk
print(sk.__version__)   
from sklearn.datasets import load_boston
import time
import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping

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
# model=Sequential()
# model.add(Dense(10, input_dim=13))
# model.add(Dense(11))
# model.add(Dense(12))
# model.add(Dense(13))
# model.add(Dense(1))

path='./_save/keras27_mcp/'
# 체크포인트로 확인
model=load_model(path+'keras27_mcp3.hdf5') 
# 세이브 모델 확인
model=load_model(path+'keras27_3_save_model.h5') #둘다 활성화해서 값나오는지 보고 밑에거만 주석처리해서 값나오는지 확인

model.summary()

# 3 컴파일, 훈련 (19_1  copy)
# model.compile(loss = 'mse', optimizer = 'adam') # 
# es = EarlyStopping(
#     monitor='val_loss',
#     mode = 'min', # 최대값 : max(accuary 측정할 때), 
#     patience=50, 
#     restore_best_weights=True,  
# )


# hist = model.fit(x_train,y_train, epochs=100, batch_size=32,
#           verbose=3,                           
#           validation_split=0.2,
#           callbacks=[es],
#          )


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