# 26-6 copy

import sklearn as sk
print(sk.__version__)   # 1.1.3
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
print(datasets.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
                              # 'B' 'LSTAT']

x=datasets.data
y=datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test=train_test_split(
    x,y, train_size=0.8, shuffle=True,
    random_state=333,
)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))   # -0.00557837618540494 1.1478180091225068

# 2 모델 구성
# model=Sequential()
# model.add(Dense(10, input_dim=13))
# model.add(Dense(11))
# model.add(Dense(12))
# model.add(Dense(13))
# model.add(Dense(1))

path='./_save/keras27_mcp/'
model=load_model(path+'keras27_mcp1.hdf5') # 모델과 가중치 모두 들어가 있음 > #2, #3 생략 가능
# model.load_weights(path+'keras26_5_save1.h5') #초기 랜덤 가중치
# model.load_weights(path+'keras26_5_save2.h5') #훈련한 가중치

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
results = model.predict(x_test)

print("loss : ", loss)
print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)
