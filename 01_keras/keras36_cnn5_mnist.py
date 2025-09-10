import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
model = Sequential()
model.add(Conv2D(64, (2,2), strides=2, input_shape=(28, 28, 1))) 
model.add(Conv2D(filters=4, kernel_size=(3,3)))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(16,)))
model.add(Dense(units=10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=20, verbose=1,
                   restore_best_weights=True,
                   )
################### mcp 세이브 파일명 만들기 ########################
import datetime
date=datetime.datetime.now()
date=date.strftime("%m%d_%H%M")

path='./_save/keras36_cnn5/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5' 
filepath=''.join([path, 'k36_', date, '_', filename])
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

#GPU
# loss :  0.0751572772860527
# acc :  0.9805999994277954
# accuracy_score :  0.9806
#걸린시간 :  126.02553009986877 초


#CPU
# loss :  0.06861071288585663
# acc :  0.9810000061988831
# accuracy_score :  0.981
#걸린시간 :  331.640150308609 초

