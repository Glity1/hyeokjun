import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping

a = np.array(range(1,101))
x_predict = np.array(range(90,106))
print(a.shape) 
# exit()
timesteps = 11

#(N, 10, 1)   계산 방법 : 총 데이터(100)-10+1 = 91 =N

# x : (N, 10, 1) -> (N, 5, 2)
# y : (N, 1)


def split_x(dataset, timesteps):                           # 이 부분 이해 확실하게 할 것.  // spli_x(dataset, timesteps) 를 정의
    aaa = []                                               # x,y를 합쳐서 했음.
    for i in range(len(dataset) - timesteps+1):           
        subset = dataset[i : (i+timesteps)]                # y에 해당하는 부분
        aaa.append(subset)                                 
        print(aaa)                                         
    return np.array(aaa)  

bbb = split_x(a, timesteps=timesteps)  
print(bbb)
print(bbb.shape)   # (90, 11)

x = bbb[: , :-1]                                            # :-1 처음부터 -2지점까지   
y = bbb[:, -1]                                              # -1 제일 마지막지점만       

print(x.shape, y.shape)                                    # (90, 10) (90,)
       
x_pred = np.array(range(96, 106))
x_pred = split_x(x_pred, timesteps-1)
print(x_pred.shape) #(1, 10)

x_pred = x_pred.reshape(x_pred.shape[0],-1,2)
print(x_pred.shape)                        # (1, 5, 2)
print(x_pred) 
# [[[ 96  97]
#   [ 98  99]
#   [100 101]
#   [102 103]
#   [104 105]]]
# exit()                  
print(x,y)
print(x.shape, y.shape)                    # (90, 10) (90,)
x = x.reshape(x.shape[0], -1, 2)  

print(x.shape)  #(90, 5, 2)

#2. 모델구성
model = Sequential()
# model.add(LSTM(16, input_shape=(5, 2), return_sequences=True,activation='relu'))         
# model.add(GRU(16, return_sequences=True, activation='relu'))
model.add(SimpleRNN(16, input_shape=(5, 2), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2400,) 

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)
y_pred = model.predict(x_pred)
print('x_predict의 결과 : ', y_pred)

# loss :  0.0030261394567787647
# x_predict의 결과 :  [[106.019806]]