import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)

x_train = x_train/255.
x_test = x_test/255.

#ohe
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)


#2. 모델
model = Sequential()
model.add(Dense(4096, input_dim=3072))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(2084, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='softmax'))

#3. 컴파일 ,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping  
es = EarlyStopping(                   
    monitor='val_loss',
    mode = 'min',                     
    patience=30,                      
    restore_best_weights=True,) 

start = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=256,    # 60000장을 64장 단위로 훈련
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es])
end = time.time()
print("걸린시간 : ", end - start, '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])  # loss
print('acc : ', loss[1])  # acc


#GPU
# loss :  3.323869228363037
# acc :  0.23000000417232513

#CPU