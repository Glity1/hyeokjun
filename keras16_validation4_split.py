# 16-3 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array(range(1, 170))
y = np.array(range(1, 170))

# [실습] train_test로 10:3:3 으로 나눈다.

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.85,
    shuffle=True,
    random_state=714,
    )

# x_train, x_val, y_train, y_val = train_test_split(
#     x_train,y_train,
#     train_size=0.8,
#     shuffle=False,
#     random_state=714,
#     )

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
# print(x_val)
# print(y_val)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 ,훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=200, batch_size=1,
          verbose=1,
          validation_split=0.2                # 머신이 훈련하면서 검증까지 가중치 갱신에 직접적인 도움 X
          )                                   # val 값을 데이터에서 고정하지말고 val_split을 써서 값을 조정하면 조금 더 편하다

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([17])

print('loss : ', loss)
print('[17]의 예측값 : ', results)
