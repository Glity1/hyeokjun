import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
x_datasets = np.array([range(100), range(301,401)]).T   # (100, 2) // 1컬럼 : 삼성전자 종가, 2컬럼 : 하이닉스 종가.

y1 = np.array(range(2001, 2101))  # (100,) // 화성의 화씨 온도.
y2 = np.array(range(13001, 13101))  # (100,) // 비트코인 가격


x_train, x_test, y_train1, y_test1, y_train2, y_test2 = train_test_split(
            x_datasets, y1, y2, 
            test_size=0.3, random_state=222)


#2-1. 모델구성
input1  = Input(shape=(2,))              # 행무시 열우선
dense1  = Dense(10, activation='relu', name='IBM1')(input1)
dense2  = Dense(20, activation='relu', name='IBM2')(dense1)
dense3  = Dense(30, activation='relu', name='IBM3')(dense2)
dense4  = Dense(40, activation='relu', name='IBM4')(dense3)
output1 = Dense(50, activation='relu', name='IBM5')(dense4)  

#2-4 분리1 -> y1
last_output11 = Dense(10, name='last11')(output1)
last_output12 = Dense(10, name='last12')(last_output11)
last_output13 = Dense(1, name='last13')(last_output12)

#2-5 분리2 -> y2
last_output21 = Dense(10, name='last21')(output1)

# 모델
model = Model(inputs =input1, outputs=[last_output13, last_output21])

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# es = EarlyStopping(
#     #monitor= 'val_loss',
#     mode= 'min',
#     patience=30,
#     restore_best_weights=True,
# )

model.fit(x_train, [y_train1, y_train2], validation_split=0.2, epochs=2400, batch_size=256)

#4. 평가, 예측
loss = model.evaluate(x_test, [y_test1, y_test2])
x1_predict = np.array([range(100,106), range(400,406)]).T
y_predict1 = model.predict(x1_predict)

print("Loss : ", loss)
print("x_test1의 예측값 : ", y_predict1)




# 출력이 2개 이상이 되는경우 각각의 loss가 있고 총 loss는 각각의 loss를 합친값이다.