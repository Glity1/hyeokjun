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
results = model.predict([x_test])

print("loss : ", loss)
print("[x]의 예측값 : ", results)

from sklearn.metrics import r2_score, mean_squared_error

# def RMSE(y_test, y_predict): #def 정의하겠다. y의 테스트 값 + 예측값 
#     return np.sqrt(mean_squared_error(y_test, y_predict)) #np.sqrt(루트를 씌운다)

#rmse = RMSE(y_test, results) #가독성을 위해 RMSE(a, b)로 적어도된다 자리만 맞추면된다.
#print('RMSE : ', rmse)

r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)


#결과
# loss :  3.444873094558716
# [x]의 예측값 :  [[ 9.269314  ]
#  [ 0.58928376]
#  [11.19821   ]
#  [ 1.5537318 ]
#  [ 5.411523  ]
#  [ 8.304867  ]]
# r2 스코어 :  0.7577823400497437 = accuracy 75%