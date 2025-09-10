from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x_train = x[0:7]
# y_train = y[0:7]

x_train = x[:7]
x_test = x[7:]

print(x_train)
print(x_test)
print(x_train.shape, x_test.shape) #(7,) (3,)
# x_train = x[0:7] 대괄호 안에 숫자는 몇번째인지를 명시하는것이다 첫번째 0 두번째 1 제일끝에서 하나 전 : -1
# x_test = x[7:10] 마지막은 -1 해준다

y_train = y[:7]
y_test = y[7:]

print(y_train)
print(y_test)
print(y_train.shape, y_test.shape) #(7,) (3,)
# y_train = x[0:7]
# y_test = x[7:10]

#[실습] 넘파이 리스트의 슬라이싱

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 ,훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=400, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([11])

print('loss : ', loss)
print('[11]의 예측값 : ', results)

# 결과
# loss :  1.5916157281026244e-12
# [11]의 예측값 :  [[11.]]
