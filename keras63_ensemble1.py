import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T   # (100, 2) // 1컬럼 : 삼성전자 종가, 2컬럼 : 하이닉스 종가.

x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose() # (100, 3) // 1컬럼 : 원유, 2컬럼 : 환율, 3컬럼 : 금시세

y = np.array(range(2001, 2101))  # (100,) // 화성의 화씨 온도.

x_train1, x_test1, x_train2, x_test2, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, test_size=0.3, random_state=222)

print(x_train1.shape, x_test1.shape)    # (70, 2) (30, 2)
print(x_train2.shape,  x_test2.shape)   # (70, 3) (30, 3)
print(y_train.shape, y_test.shape)      # (70,) (30,)

#2-1. 모델구성
input1  = Input(shape=(2,))              # 행무시 열우선
dense1  = Dense(10, activation='relu', name='IBM1')(input1)
dense2  = Dense(20, activation='relu', name='IBM2')(dense1)
dense3  = Dense(30, activation='relu', name='IBM3')(dense2)
dense4  = Dense(40, activation='relu', name='IBM4')(dense3)
output1 = Dense(50, activation='relu', name='IBM5')(dense4)   # 앙상블에서 output은 마음대로 잡아도된다 왜냐면 2모델이 합쳐질 output이 따로 또 있기때문이다.
# model1  = Model(inputs=input1, outputs=output1)             # 어차피 마지막에 model 구성함
# model1.summary()

#2-2 모델구성
input2  = Input(shape=(3,))
dense21 = Dense(100, activation='relu', name='IBM21')(input2)
dense22 = Dense(50, activation='relu', name='IBM22')(dense21)
output2 = Dense(30, activation='relu', name='IBM23')(dense22)
# model2  = Model(inputs=input2, outputs=output2)
# model2.summary()

#2-3 모델 합치기 // 두개를 합치는 레이어를 구성해서 합친다
from keras.layers.merge import concatenate, Concatenate  # 소문자 함수 // 대문자 클래스
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(40, activation='relu', name='mg2')(merge1)
merge3 = Dense(20, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()

# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 2)]          0           []

#  IBM1 (Dense)                   (None, 10)           30          ['input_1[0][0]']

#  IBM2 (Dense)                   (None, 20)           220         ['IBM1[0][0]']

#  input_2 (InputLayer)           [(None, 3)]          0           []

#  IBM3 (Dense)                   (None, 30)           630         ['IBM2[0][0]']

#  IBM21 (Dense)                  (None, 100)          400         ['input_2[0][0]']

#  IBM4 (Dense)                   (None, 40)           1240        ['IBM3[0][0]']

#  IBM22 (Dense)                  (None, 50)           5050        ['IBM21[0][0]']

#  IBM5 (Dense)                   (None, 50)           2050        ['IBM4[0][0]']

#  IBM23 (Dense)                  (None, 30)           1530        ['IBM22[0][0]']

#  mg1 (Concatenate)              (None, 80)           0           ['IBM5[0][0]',
#                                                                   'IBM23[0][0]']

#  mg2 (Dense)                    (None, 40)           3240        ['mg1[0][0]']

#  mg3 (Dense)                    (None, 20)           820         ['mg2[0][0]']

#  last (Dense)                   (None, 1)            21          ['mg3[0][0]']

# ==================================================================================================
# Total params: 15,231
# Trainable params: 15,231
# Non-trainable params: 0
# __________________________________________________________________________________________________

# 엄밀하게 말하면 이것도 단일모델이다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    patience=30,
    restore_best_weights=True,
)

model.fit([x_train1, x_train2], y_train, validation_split=0.2, epochs=2400, batch_size=256)

#4. 평가, 예측
loss = model.evaluate([x_test1, x_test2], y_test)
x1_predict = np.array([range(100,106), range(400,406)]).T
x2_predict = np.array([range(200,206), range(510,516), range(249, 255)]).T
y_predict = model.predict([x1_predict, x2_predict])

print("Loss : ", loss)
print("[x_test1, x_test2]의 예측값 : ", y_predict)

# y_pred 는 2101  부터 2106까지 나오면 된다.


# Loss :  0.0246613509953022
# [x_test1, x_test2]의 예측값 :  [[2097.6343][2101.8955][2106.1577][2110.4197][2114.682 ][2118.9438]]