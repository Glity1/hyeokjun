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

print("스케일링 전 x_test의 최대값:", np.max(x_test))
print("스케일링 전 x_test의 최소값:", np.min(x_test))


########## 스케일링 1번 MinMaxScaler() ############
x_train = x_train.reshape(60000, 28*28)                                            # 다양하게 쓰는방법 이해하기 위아래 똑같은 말이다.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
print(x_train.shape, x_test.shape)    # (60000, 784), (10000, 784)

scaler = StandardScaler()               # sklearn 은 최소한 행렬이나 벡터 형태의 데이터를 받는다
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 24.0 0.0   한 컬럼에서 최대값이 255가 아니면 분모자체가 255보다 작아지는데 분자는 커지니까 24가 나올 수 있음.



# 스케일링 2번 ######### 0~1사이로 변환 : 정규화 (많이 쓴다)
# x_train = x_train/255. #255에서 . 을 붙이면 소수점 연산을 하란말이다.
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

# 스케일링 3번 ######### 127.5 지점을 중간지점을 0으로 잡고 -1과1사이로 바꾼다 : 정규화2 (많이 쓴다)
# x_train = (x_train -127.5) / 127.5 
# x_test = (x_test -127.5) / 127.5 
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

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
model.add(Conv2D(64, (2,2), strides=2, input_shape=(28, 28, 1))) 
model.add(Conv2D(filters=4, kernel_size=(2,2)))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
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


# 1번-1 MinMaxscaler()
# Epoch 00031: early stopping
# 걸린시간 :  112.76881814002991 초
# loss :  0.055788446217775345
# acc :  0.9840999841690063
# accuracy_score :  0.9841

# 1번-2 standardscaler()
# 걸린시간 :  108.36879920959473 초
# loss :  0.06128533184528351
# acc :  0.9832000136375427
# accuracy_score :  0.9832

# 2번 /255
# Epoch 00034: early stopping
# 걸린시간 :  113.56311988830566 초
# loss :  0.06070977821946144
# acc :  0.9818000197410583
# accuracy_score :  0.9818


# 3번 /127.5
# Epoch 00035: early stopping
# 걸린시간 :  112.41223001480103 초
# loss :  0.059207942336797714
# acc :  0.984000027179718
# accuracy_score :  0.984
