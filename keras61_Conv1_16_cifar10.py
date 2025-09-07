import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, BatchNormalization, Dropout, Flatten, Conv1D, MaxPooling2D, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)
 
x_train = x_train/255.
x_test = x_test/255.

y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000,)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape) #(50000, 10) (10000, 10)

x_train = x_train.reshape(x_train.shape[0], 32*32, 3)
x_test = x_test.reshape(x_test.shape[0], 32*32, 3)

# exit()
#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, input_shape=(32*32, 3), padding='same'))         
model.add(Flatten())
model.add(Activation('relu')) 
model.add(Dropout(0.1))
model.add(Dense(units=10, activation='softmax'))

#3. 컴파일 ,훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode = 'min',
                   patience=1, verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit(x_train, y_train, epochs=10, batch_size=64,    # 60000장을 64장 단위로 훈련
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

# loss :  0.6186747550964355
# acc :  0.7900999784469604
# accuracy_score :  0.7901

