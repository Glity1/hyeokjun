from tensorflow.keras.models import Sequential #대문자 클래스
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.model_selection import train_test_split # 소문자 함수

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습] train과 test를 섞어서 sklearn으로 7:3으로 나눠라.

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, # 디폴트 : 0.75
                                                    # test_size=0.3, # 생략가능 디폴트 : 0.25
                                                    shuffle=True,  # 디폴트 : True
                                                    random_state=121
                                                    ) #랜덤난수: 121번 난수에 랜덤데이터를 고정시켰다

print(x_train)
print(x_test)
print(y_train)
print(y_test)

#exit()

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
