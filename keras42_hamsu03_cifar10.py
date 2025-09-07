import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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

print(y_train)
# [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 1.]
#  [0. 0. 0. ... 0. 0. 1.]
#  ...
#  [0. 0. 0. ... 0. 0. 1.]
#  [0. 1. 0. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]]
# exit()

#2. 모델
input1 = Input(shape=(3072,))       # 대문자 클래스 를 인스턴스화 한다.
dense1 = Dense(10, name='layer_1')(input1)       # (input) = input 에서 받아들인다 
drop1 = Dropout(0.1)(dense1)
dense2 = Dense(11, name='layer_2', activation='relu')(drop1)        # name='원하는 이름 ' : summary에서 보이는 레이어의 이름을 지을 수 있음
drop2 = Dropout(0.1)(dense2)
dense3 = Dense(12, name='layer_3', activation='relu')(drop2)         # 임의적으로 순서를 바꿔서 모델을 구성 가능하다.
drop3 = Dropout(0.1)(dense3)
dense4 = Dense(13, name='layer_4', activation='relu')(drop3)
drop4 = Dropout(0.1)(dense4)
output1 = Dense(10, activation='softmax')(drop4)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일 ,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping  
es = EarlyStopping(                   
    monitor='val_loss',
    mode = 'min',                     
    patience=20,                      
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

# loss :  1.5022348165512085
# acc :  0.4846999943256378