# 08-1 copy 해서 가져옴  // validation(확인, 검증)

###################validation의 사용법#############################

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) # (10,)
# print(y.shape) # (10,)

x_train = np.array([1,2,3,4,5,6,])          # 훈련 데이터
y_train = np.array([1,2,3,4,5,6,])

x_val = np.array([7,8])                     # 훈련에 대한 검증을 위한 데이터
y_val = np.array([7,8])

x_test = np.array([9,10])                   # 평가 데이터
y_test = np.array([9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 ,훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=400, batch_size=1,
          validation_data=(x_val, y_val)                # 머신이 훈련하면서 검증까지 가중치 갱신에 직접적인 도움 X
          )


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([11])

print('loss : ', loss)
print('[11]의 예측값 : ', results)


# val_loss 의 데이터가 중요함. 
# loss 가 내려가다가 val_loss가 정체되면 학습이 잘 안되는중이다 앞으로 성능이 좋아지는 기준은 val_loss로 본다.
# loss 가 좋아지진않지만 val_loss가 좋아질 경우 성능이 좋아진다고 본다.
# val은 train 보다는 적게 잡으면 되지만 val자체를 너무 많이 잡을필요는없다.
# data를 나눌때 8:2, 7:3 정도로 보통 나눈다.