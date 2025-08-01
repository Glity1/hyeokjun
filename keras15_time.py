# 14 copy 해서 가져옴
######################### 시간에 대한 모듈 사용방법에 대해서 #############################

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import time  # 시간에 대한 모듈 import

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) # (10,)
# print(y.shape) # (10,)

x_train = np.array(range(100))
y_train = np.array(range(100))

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
start_time = time.time()                            # time(pyhthon에서 제공하는 타임패키지를 사용). time(현재시간) :  현재 시간을 반환, 시작시간.
print(start_time)                                   # 1747968519.6089365 // 특정기준 시간에서 소요된 시간
model.fit(x_train,y_train, epochs=1000, batch_size=150, 
          verbose=1  
          )
                        # 출력을 줄여준다 
                        # 0일때 출력이 거의없다 
                        # 1일때 default 
                        # 2일때 progress bar가 없어진다 /(진행도를 볼수없음.) 
                        # /0,1,2제외 epoch 값만
end_time = time.time()           
print("걸린시간 : ", end_time - start_time, '초')

# 한번 훈련 돌릴때 걸리는 시간을 파악해야지 제출하는 시간까지의 시간을 고려할수있다.
# 효율적인 훈련을 위해 파악 중요함.

#1) epochs = 1000 에서 verbos = 0,1,2,3의 시간을 적는다.
    # verbos = 0 / 걸린시간 :  45.157942056655884 초
    # verbos = 1 / 걸린시간 :  53.62514591217041  초
    # verbos = 2 / 걸린시간 :  41.881927490234375 초
    # verbos = 3 / 걸린시간 :  42.906530141830444 초

#2) epochs = 1000 에서 verbos = 1의 시간을 적는다. - batch 1, 32, 128 일때 시간.

    # batch = 1   / 걸린시간 : 57.2393798828125   초
    # batch = 32  / 걸린시간 : 3.288198232650757  초
    # batch = 128 / 걸린시간 : 1.665989875793457  초
    # batch = 150 / 걸린시간 : 1.6359350681304932 초
# exit()       
# #4. 평가, 예측
# loss = model.evaluate(x_test,y_test)
# results = model.predict([11])

# print('loss : ', loss)
# print('[11]의 예측값 : ', results)

# # 결과
# loss :  1.5916157281026244e-12
# [11]의 예측값 :  [[11.]]
# 사람이 이 출력을 보는데 0.001초 걸리고 기계는 이걸 계산하는데 0.0000001초 걸린다 근대 사람한테 보여줄려고 일부러 사람이 보게 하는 시간으로 딜레이시킨다.
# 우리가 볼만한게 여러가지가 있다 loss 하나만 보는중
