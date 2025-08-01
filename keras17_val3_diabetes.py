import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes                               #당뇨병 diabetes
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.1,
    random_state=814 # validation_split로도 바뀌지않는다면 바꾸자
)

#2. 모델구성
model = Sequential()
model.add(Dense (40, input_dim=10))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=200, batch_size=1,
          verbose =1,
          validation_split=0.25)

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
results = model.predict([x_test])

print('loss : ', loss)
print("[x_test]의 예측값 : ", results)

r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)



# 목표 r2 0.62 이상
# loss :  2281.981201171875
# r2 스코어 :  0.6295709504734388 epchos=1000   layer를 원상태유지

# 데이터 크기에 맞는 epcohs 와 layer의 길이가 있음 적절히 맞출것 