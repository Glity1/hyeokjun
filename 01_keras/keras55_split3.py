import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# 101~107 을 찾기
a = np.array(range(1,101))
timesteps = 6
print(a.shape) # (100,)


def split_x(dataset, timesteps):                           # 이 부분 이해 확실하게 할 것.  // spli_x(dataset, timesteps) 를 정의
    aaa = []                                               # x,y를 합쳐서 했음.
    for i in range(len(dataset) - timesteps+1):           
        subset = dataset[i : (i+timesteps)]                # y에 해당하는 부분
        aaa.append(subset)                                 
        print(aaa)                                         
    return np.array(aaa)                                                                 

bbb = split_x(a, timesteps=timesteps)  
print(bbb)
print(bbb.shape)   # (95, 6)  
  
x = bbb[: , :-1]                                            # :-1 처음부터 -2지점까지   
y = bbb[:, -1]                                              # -1 제일 마지막지점만       
       
x_pred = np.array(range(96, 106))
x_pred = split_x(x_pred, timesteps-1)
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1)
print(x_pred.shape)
                      
print(x,y)
print(x.shape, y.shape)                    # (95, 5) (95,)
x = x.reshape(x.shape[0], x.shape[1], 1)  
print(x.shape)  #(95, 5, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(16, input_shape=(5, 1), return_sequences=True,activation='relu'))         
model.add(GRU(16, return_sequences=True, activation='relu'))
model.add(SimpleRNN(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2500,) 

#4. 평가, 예측
results = model.evaluate(x, y)
print("loss : ", results)
y_pred = model.predict(x_pred)
print('range[96,106]의 결과 : ', y_pred)

# loss :  0.00029809228726662695
# range[96,106]의 결과 :  
# [[101.01367 ]
#  [102.053734]
#  [103.09509 ]
#  [104.13778 ]
#  [105.181885]
#  [106.227425]]


