# 10-2 copy

#1. R2를 음수가 아닌 0.4 이하로 만들것.
#2. 데이터는 건들지 말것
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size=1 고정
#5. 히든레이어의 노드는 10개 이상 100개 이하
#6. train 사이즈 75%
#7. epoch 100번 이상
#8. loss지표는 mse
#9. dropout 넣지 말것 // 250611 추가

# [실습시작]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    random_state=250
)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(12, activation='linear'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= Adam(learning_rate=1e-5))
model.fit(x_train, y_train, epochs=100, batch_size=1)

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

# loss :  35.38602828979492

# r2 스코어 :  0.14111602012432511

# loss :  10.988797187805176

# r2 스코어 :  0.09631608608219566