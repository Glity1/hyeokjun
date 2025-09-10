import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000,)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)


#2. 모델구성

model = Sequential()
model.add(Conv2D(30000, (2,2), strides=1, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(30000, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Conv2D(1024, (2,2), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(1024, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(Dropout(0.1))

model.add(Conv2D(1024, (2,2), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(1024, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='softmax'))

#3. 컴파일 ,훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode = 'min',
                   patience=100, verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit(x_train, y_train, epochs=5000, batch_size=1,    # 60000장을 64장 단위로 훈련
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
