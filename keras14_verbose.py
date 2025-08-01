# 08-1 copy 해서 가져옴
# 이 훈련에 문제점은? 1. 출력되는게 많다 / 적다 둘다 맞음  2. 출력하는 것의 문제점? 시간이 오래걸린다
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) # (10,)
# print(y.shape) # (10,)

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 ,훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=400, batch_size=1, 
          verbose=0  
          )
                        # 출력을 줄여준다 
                        # 0일때 출력이 거의없다 
                        # 1일때 default 
                        # 2일때 progress bar가 없어진다 /(진행도를 볼수없음.) 
                        # /0,1,2제외 epochs 값만
                    
#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([11])

print('loss : ', loss)
print('[11]의 예측값 : ', results)

# 결과
# loss :  1.5916157281026244e-12
# [11]의 예측값 :  [[11.]]
# 사람이 이 출력을 보는데 0.001초 걸리고 기계는 이걸 계산하는데 0.0000001초 걸린다 근대 사람한테 보여줄려고 일부러 사람이 보게 하는 시간으로 딜레이시킨다.
# 우리가 볼만한게 여러가지가 있다 loss 하나만 보는중
