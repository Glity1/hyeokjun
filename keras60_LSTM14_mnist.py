# CNN 을 DNN으로 만든다

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM
import time
from sklearn. metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

# 스케일링
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

print(x_train.shape[0]) # 60000  행 
print(x_train.shape[1]) # 28  가로
print(x_train.shape[2]) # 28  세로
# print(x_train.shape[3]) # 에러  이유 : 현재 형태가 (60000,28,28)

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])


#ohe
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)  # y는 현재 벡터형태 (60000,) 필요한건 매트릭스 형태로 변경 필요 즉 reshape 을 통해 (60000,1)로 변경하자

# reshape 
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)   # -1 의 의미 데이터의 가장 마지막,끝의 값 ex) 1만번째값을 가져온다
print(y_train.shape, y_test.shape) #(60000, 1) (10000, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델
model = Sequential()
model.add(LSTM(10, input_shape=(784, 1),activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 ,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping  
es = EarlyStopping(                   
    monitor='val_loss',
    mode = 'min',                     
    patience=50,                      
    restore_best_weights=True,) 

start = time.time()
hist = model.fit(x_train, y_train, epochs=1, batch_size=128,    # 60000장을 64장 단위로 훈련
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es])
end = time.time()
print("걸린시간 : ", end - start, '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])  # loss
print('acc : ', loss[1])  # acc


# CPU 가 좀 더 빠름
# 걸린시간 :  85.76393675804138 초
# loss :  0.08059024810791016
# acc :  0.9824000000953674