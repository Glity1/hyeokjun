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
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
sampleSubmission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop(['target'], axis=1)
y = train_csv['target'] 

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=74,
    stratify=y
    )


scaler=StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_train)
test_csv_scaled = scaler.transform(test_csv)

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