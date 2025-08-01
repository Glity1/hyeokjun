from tensorflow.keras.models import Sequential #Sequential 순차적으로간다
from tensorflow.keras.layers import Dense #레이어를 댄스(밀집형태)로 구성한다
import numpy as np 

#1 데이터
x= np.array([1,2,3,4,5,6]) # numpy형태로 묶어서 던져준다 dim 1개는 리스트 1개의 차원
y= np.array([1,2,3,5,4,6])

# epchs는 100으로 고정
# loss 기준 0.32 미만으로 만들것.


#2 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1)) 
model.add(Dense(300)) 
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(1)) 

epochs=100
#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=epochs)

#4 평가, 예측
loss = model.evaluate(x,y)
print("##################################################")
print("epochs : ", epochs)
print("loss : ", loss)
# results = model.predict([7])
# print('6의 예측값 : ', results)


##################################################
#epochs :  100
#loss :  0.3239242732524872