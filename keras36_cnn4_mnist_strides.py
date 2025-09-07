import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

# x를 reshape -> (60000,28,28,1)로 변경
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape, x_test.shape)    # (60000, 28, 28, 1) (10000, 28, 28, 1)


# ohe 적용
y_train = pd.get_dummies(y_train)     # (60000, 10)
y_test = pd.get_dummies(y_test)       # (10000, 10)
print(y_train.shape, y_test.shape)

#2. 모델구성
# model = Sequential()
# model.add(Conv2D(5, (3,3), strides=2, input_shape=(10, 10, 1)))   # output shape (a,b,c)에서 a,b 구하는 방법 = inpurt_shape (a,b) - kernel_size(a,b) 구한 값에 + 1 씩
# model.add(Conv2D(filters=4, kernel_size=(2,2)))
# model.add(Conv2D(16, (3,3)))
# model.add(Flatten())
# model.add(Dense(units=10))
# model.add(Dense(units=16))
# model.add(Dense(units=10, activation='softmax'))
# model.summary()

#2. 모델구성
model = Sequential()
model.add(Conv2D(5, (3,3), strides=2, input_shape=(28, 28, 1)))   # output shape (a,b,c)에서 a,b 구하는 방법 = inpurt_shape (a,b) - kernel_size(a,b) 구한 값에 + 1 씩
model.add(Conv2D(filters=4, kernel_size=(2,2)))
model.add(Conv2D(3, (3,3)))
model.add(Flatten())
model.add(Dense(units=16))
model.add(Dense(units=16))
model.add(Dense(units=10, activation='softmax'))
model.summary()










