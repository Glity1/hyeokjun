import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=333
)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1)

#4. 평가, 예측
print("=====================================================")
loss = model.evaluate(x_test, y_test)
results = model. predict([x])

print("loss : ", loss)
print("[x]의 예측값 : ", results)

#그래프
import matplotlib.pyplot as plt #matplotlib (그래프를 그림)
plt.scatter(x,y) #데이터 점 찍기 scatter (흩뿌리다) x,y에 점하나 찍는다
plt.plot(x, results) # plot 선 긋는다
plt.plot(x, results, color= 'red')
plt.show()

#결과
