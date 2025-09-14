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
path = './_data/jena/'
dataset_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
dataset_csv.index = pd.to_datetime(dataset_csv.index)

# 입력 데이터 x, 타겟 데이터 y 생성
x = dataset_csv[['wv (m/s)', 'max. wv (m/s)', 'T (degC)']]
y = dataset_csv['wd (deg)']


# MinMaxScaler 적용
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# 2. 시퀀스 데이터 생성 함수 정의
def split_xy(x, y, timesteps, target_steps, stride):
    x_seq, y_seq = [], []
    for i in range(0, len(x) - timesteps - target_steps + 1, stride):
        x_seq.append(x[i : i + timesteps])
        y_seq.append(y[i + timesteps : i + timesteps + target_steps])
    return np.array(x_seq), np.array(y_seq)

# 3. 시퀀스 구성 
timesteps = 144
target_steps = 144
stride = 1
x_seq, y_seq = split_xy(x_scaled, y.to_numpy(), timesteps, target_steps, stride)

# 4. 학습/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_seq, y_seq, test_size=0.2, random_state=222, shuffle=True
)

optim = [Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]
lr = [0.1, 0.01, 0.001, 0.0001]  # 학습률을 바꿔가면서 돌려보자]



#2. 모델구성
model = Sequential()
model.add(Dense(128,input_dim=93, activation='relu'))
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

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_test, results)
R2 = r2_score(y_test, results)

print("rmse : ", rmse)
print('r2 : ', R2)

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


