#26-5 copy
#_save 폴더 하위> keras27_mcp 폴더 만듦

import sklearn as sk
print(sk.__version__)   # 1.1.3
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
    patience=10,
    restore_best_weights=True,
    verbose=1       # 가장 좋은 가중치의 결과, default=True
)

path='./_save/keras27_mcp/'
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    #가장 좋은 값 저장
    filepath=path+'keras27_mcp1.hdf5'   #확장자 hdf5, h5 상관x
)

hist = model.fit(x_train,y_train, epochs=300, batch_size=32, 
          verbose=1,                          
          validation_split=0.2,
          callbacks=[es, mcp],  # mcp 삽입
          )

# path='./_save/keras26/'
# model.save(path+'keras26_5_save2.h5')

#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict([x_test])

print("loss : ", loss)
# print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)
