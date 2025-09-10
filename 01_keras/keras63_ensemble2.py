import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T   # (100, 2) // 1컬럼 : 삼성전자 종가, 2컬럼 : 하이닉스 종가.

x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose() # (100, 3) // 1컬럼 : 원유, 2컬럼 : 환율, 3컬럼 : 금시세

x3_datasets = np.array([range(100), range(301, 401), range(77, 177), range(33, 133)]).transpose() # (100, 3) // 1컬럼 : 원유, 2컬럼 : 환율, 3컬럼 : 금시세

y = np.array(range(2001, 2101))  # (100,) // 화성의 화씨 온도.

x_train1, x_test1, x_train2, x_test2, x_train3, x_test3, y_train, y_test = train_test_split(x1_datasets, x2_datasets, x3_datasets, y, test_size=0.3, random_state=222)

# scaler1 = MinMaxScaler()
# scaler2 = MinMaxScaler()
# scaler3 = MinMaxScaler()

# x_train1 = scaler1.fit_transform(x_train1)
# x_test1  = scaler1.transform(x_test1)

# x_train2 = scaler2.fit_transform(x_train2)
# x_test2  = scaler2.transform(x_test2)

# x_train3 = scaler3.fit_transform(x_train3)
# x_test3  = scaler3.transform(x_test3)

print(x_train1.shape, x_test1.shape)    # (70, 2) (30, 2)
print(x_train2.shape,  x_test2.shape)   # (70, 3) (30, 3)
print(x_train3.shape,  x_test3.shape)   # (70, 4) (30, 4)
print(y_train.shape, y_test.shape)      # (70,) (30,)

# exit()
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

#2-2 모델구성
input3  = Input(shape=(4,))
dense31 = Dense(80, activation='relu', name='IBM31')(input3)
dense32 = Dense(40, activation='relu', name='IBM32')(dense31)
output3 = Dense(20, activation='relu', name='IBM33')(dense32)
# model3  = Model(inputs=input3, outputs=output3)
# model3.summary()


#2-3 모델 합치기 // 두개를 합치는 레이어를 구성해서 합친다
from keras.layers.merge import concatenate, Concatenate  # 소문자 함수 // 대문자 클래스
merge1 = Concatenate(name='mg1')([output1, output2, output3])
merge2 = Dense(40, activation='relu', name='mg2')(merge1)
merge3 = Dense(20, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
model = Model(inputs=[input1, input2, input3], outputs=last_output)
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    patience=30,
    restore_best_weights=True,
)

model.fit([x_train1, x_train2, x_train3], y_train, validation_split=0.2, epochs=2400, batch_size=256)

#4. 평가, 예측
loss = model.evaluate([x_test1, x_test2, x_test3], y_test)
x1_predict = np.array([range(100,106), range(400,406)]).T
x2_predict = np.array([range(200,206), range(510,516), range(249, 255)]).T
x3_predict = np.array([range(100,106), range(400,406), range(177,183), range(133, 139)]).T
y_predict = model.predict([x1_predict, x2_predict, x3_predict])

print("Loss : ", loss)
print("[x_test1, x_test2]의 예측값 : ", y_predict)

# y_pred 는 2101  부터 2106까지 나오면 된다.


# 