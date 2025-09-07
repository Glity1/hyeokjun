# 36_3 copy

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Conv1D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score


#1. 데이터
(x_train, y_train), (x_test, y_test)= fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)      # 총 7만장

print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
print(pd.value_counts(y_test))

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

column_names = ohe.get_feature_names()

x_train = x_train.reshape(x_train.shape[0], 28*28, 1)
x_test = x_test.reshape(x_test.shape[0], 28*28, 1)
    
# exit()
#2. 모델
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(28*28, 1), padding='same', activation='relu'))         
model.add(Flatten())
model.add(Dense(256, input_dim=784))
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
hist = model.fit(x_train, y_train, epochs=1, batch_size=256,    # 60000장을 64장 단위로 훈련
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es])
end = time.time()
print("걸린시간 : ", end - start, '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])  # loss
print('acc : ', loss[1])  # acc


































