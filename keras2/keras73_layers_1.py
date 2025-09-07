# 72-2 copy

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D
import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import GlobalAveragePooling2D


SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)
 
x_train = x_train/255.
x_test = x_test/255.

y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000,)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
# print(y_train.shape, y_test.shape) #(50000, 10) (10000, 10)

# x_train = x_train.reshape(x_train.shape[0], 32*32, 3)
# x_test = x_test.reshape(x_test.shape[0], 32*32, 3)

vgg16 = VGG16(
    include_top=False,
    input_shape=(32, 32, 3), 
    # pooling='avg'
    )

vgg16.trainable = True

model = Sequential()
model.add(vgg16)      # 26
# model.add(GlobalAveragePooling2D()) # 파라미터로 가능
model.add(Flatten())
model.add(Dense(100))  #2
model.add(Dense(10, activation='softmax'))   #2

model.summary()

print(len(model.weights))  #30
print(len(model.trainable_weights)) #4  동결된 vgg16 26개 훈련불가

# trainable true : 30 30
# trainable False : 30 4
exit()

#3. 컴파일 ,훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode = 'min',
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit(x_train, y_train, epochs=100, batch_size=64,    # 60000장을 64장 단위로 훈련
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])  # loss
print('acc : ', loss[1])  # acc

y_pred = model.predict(x_test)
y_test = y_test.values
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)  

acc= accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)

