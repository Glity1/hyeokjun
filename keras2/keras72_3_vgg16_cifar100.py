import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, BatchNormalization, Dropout
import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import GlobalAveragePooling2D

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

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
    pooling='avg'
    )

vgg16.trainable = True   

model = Sequential()
model.add(vgg16)
# model.add(GlobalAveragePooling2D()) # 파라미터에서 사용
# model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

model.summary()

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

# 실습 # 
# 비교할것
# 1. 이전의 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False

# Flatten 과 GAP 비교

#데이터 하나당 4개 비교

# 이전결과
# loss :  0.6186747550964355
# acc :  0.7900999784469604
# accuracy_score :  0.7901

# 동결
# loss :  1.2354717254638672
# acc :  0.5709999799728394
# accuracy_score :  0.571

# 비동결
# loss :  2.303495407104492
# acc :  0.10000000149011612
# accuracy_score :  0.1

# Flatten 없음

# GAP 사용