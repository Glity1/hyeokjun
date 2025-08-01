import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=333
)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측
print("=======================================================")
loss = model.evaluate(x_test,y_test)
results = model.predict([x])

print("loss : ", loss)
print("[x]의 예측값 : ", results)

#그래프
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x, results)
plt.plot(x, results, color = 'red')
plt.show()

#결과
# loss :  3.401322603225708
# [x]의 예측값 :  [[ 0.65172416]
#  [ 1.6096743 ]
#  [ 2.567624  ]
#  [ 3.5255747 ]
#  [ 4.4835243 ]
#  [ 5.4414744 ]
#  [ 6.399425  ]
#  [ 7.357374  ]
#  [ 8.315325  ]
#  [ 9.273274  ]
#  [10.231226  ]
#  [11.189173  ]
#  [12.147126  ]
#  [13.105075  ]
#  [14.063023  ]
#  [15.020972  ]
#  [15.978924  ]
#  [16.936872  ]
#  [17.894821  ]
#  [18.852774  ]]
