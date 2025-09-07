#36-6 copy

import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)


# 스케일링 2번 ######### 0~1사이로 변환 : 정규화 (많이 쓴다)
x_train = x_train/255. #255에서 . 을 붙이면 소수점 연산을 하란말이다.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

# x를 reshape -> (60000,28,28,1)로 변경
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape, x_test.shape)    # (60000, 28, 28, 1) (10000, 28, 28, 1)

#ohe 적용
y_train = pd.get_dummies(y_train)     # (60000, 10)
y_test = pd.get_dummies(y_test)       # (10000, 10)
print(y_train.shape, y_test.shape)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), strides=1, input_shape=(28, 28, 1))) 
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

# exit()
#3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=20, verbose=1,
                   restore_best_weights=True,
                   )
################### mcp 세이브 파일명 만들기 ########################
# import datetime
# date=datetime.datetime.now()
# date=date.strftime("%m%d_%H%M")

path='./_save/keras37/'
filename='.hdf5' 
# filepath=''.join([path, 'k36_', date, '_', filename])
filepath=''.join([path, 'k37_0', filename])
###################################################################
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=filepath   
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=64,    # 60000장을 64장 단위로 훈련
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp])
end = time.time()
print("걸린시간 : ", end - start, '초')

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

# MaxPooling 적용 후 

# Epoch 00043: early stopping
# 걸린시간 :  185.2695345878601 초
# loss :  0.043674055486917496
# acc :  0.991100013256073
# accuracy_score :  0.9911
