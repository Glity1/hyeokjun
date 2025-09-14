from keras.models import Sequential, load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#1. 데이터
path = './_data/kaggle/bike/' 

train_csv = pd.read_csv(path + 'train.csv', index_col=0) # datetime column을 index 로 변경 아래 세줄 동일
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
samplesubmission_csv = pd.read_csv(path + 'samplesubmission.csv')

x = train_csv.drop(['count', 'casual', 'registered'], axis=1) 

y = train_csv['count'] 

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=74, # validation_split로도 바뀌지않는다면 바꾸자
    )

optim = [Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]
lr = [0.1, 0.01, 0.001, 0.0001]  # 학습률을 바꿔가면서 돌려보자]



#2. 모델구성
model = Sequential()
model.add(Dense(128,input_dim=8, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(BatchNormalization())
model.add(Dense(1))

epochs = 100

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

ES = EarlyStopping(monitor='val_loss',
                    mode= 'min',
                    patience= 20, verbose=1,
                    restore_best_weights= True)

rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                        patience=20, verbose=1,
                        factor=0.5,
                        )

# 0.1 / 0.05 / 0.025 / 0.0125 / 0.00625 #### 0.5
# 0.1 / 0.01 / 0.001 / 0.0001 / 0.00001 #### 0.5
# 0.1 / 0.09 / 0.081 / 0.0729 / ...     #### 0.5

import time

start = time.time()
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
        verbose=2,
        validation_split=0.2,
        #   callbacks= [ES,MCP]
        )
end = time.time()
# path = './practice/_save/'

# model.save(path + '20250531_02_california.h5')

''' 20250531 갱신
loss : 0.28095752000808716
rmse : 0.5300542568053527
R2 : 0.792117619711086

[MCP]
loss : 0.2854580879211426
rmse : 0.5342827826116628
R2 : 0.788787612383781

[load]

'''

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, results)
R2 = r2_score(y_test, results)

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

if gpus:
    print('GPU 있다!!!')
else:
    print('GPU 없다...')

time = end - start
print("소요시간 :", time)


'''
GPU 있다!!!
소요시간 : 279.6294457912445

GPU 없다...
소요시간 : 45.43129324913025

'''