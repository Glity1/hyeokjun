# https://dacon.io/competitions/open/235576/overview/description

import numpy as np                                           # 훈련에 특화됨.
import pandas as pd                                          # 데이터분석을 할 때 전처리 정제에서 유명함.
print(np.__version__)                                        # 1.23.0
print(pd.__version__)                                        # 2.2.3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/dacon/따릉이/'      # .(점 한개) = 현재 작업폴더 study25

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # a=b b를 a에 넣겠다. // index_col : 이 컬럼은 인덱스다
print(train_csv)                    # [1459 rows x 10 columns] = [1459,10]                  

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(test_csv)                     # [715 rows x 9 columns] = [715,9]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(submission_csv)               # [715 rows x 1 columns] = [715,9]  // Nan 데이터가 없음 : 결측치

print(train_csv.shape)              #(1459,10)
print(test_csv.shape)               #(715,9)
print(submission_csv.shape)         #(715,1)

print(train_csv.columns)
                                    # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                                    #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                                    #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                                    #       dtype='object')
print(train_csv.info())             # 결측치 확인가능
                                    # 총 1459개인데 컬럼마다 데이터가 빠져있다 데이터를 삭제하면 일부만 없어지더라도 그 행 가로줄이 다 삭제해야한다.
                                    # 최악의 경우 1459에서 하나도 안겹친 데이터가 다 빠져서 많은 데이터가 날라간다.
                                    # 빠진 데이터 위아래값하고 같게 하거나 평균값으로 데이터를 채워도 괜찮다
                                    # 하지만 실제 데이터하고 다를 수 있음 주의 요망 // 결측치 처리에 대한 생각을 많이 해야함.
                                    # 결측치 삭제 or 수정
print(train_csv.describe())

################################ 결측치 처리 2. 평균값 #####################################

# 결측치 자리 위치를 파악해서 특정값을 넣기
train_csv = train_csv.fillna(train_csv.mean())    # train_csv에 평균값을 결측치 자리에 넣겠다 (컬럼별 평균) / nan 자리에 컬럼 한 열의 평균값이 다 동일하게 들어간다.
print(train_csv.isna().sum())
print(train_csv.info())

################################ 테스트 데이터에 결측치 #####################################
print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

#################### 받은 데이터를 전처리 (결측치 정리) ######################

#################### x,y 데이터를 분리한다 ##########################

x = train_csv.drop(['count'], axis=1) # 축1번 count 컬럼을 삭제하고 남은 데이터를 x에 넣는다
                                      # count 라는 axis=1열 삭제, 참고로 행은 axis=0 (2차원)
                                      # train_csv에서 9개 컬럼만 복사해서 x에 넣어준거지 train_csv의 상태는 그대로다
                                      # x= count 제외한 데이터 y= count 데이터

print(x)                              # [1459 rows x 9 columns]                                

y = train_csv['count']               # count컬럼만 빼서 y에 넣겠다.

print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=74 # validation_split로도 바뀌지않는다면 바꾸자
    )

#2. 모델구성
# model = Sequential()
# model.add(Dense(30, input_dim=9))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(1))

model = Sequential()
model.add(Dense(256, input_dim=9))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs=300, batch_size=32,
          verbose =1,
          validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict(x_test)

print("loss : ", loss)
r2 = r2_score(y_test, results)
print('r2 : ', r2)

def RMSE(y_test, results):
    return np.sqrt(mean_squared_error(y_test, results)) 
rmse = RMSE(y_test, results) 
print('RMSE : ', rmse)

"""
목표
r2 0.58 이상
 loss 2400이하

 loss :  2149.123046875
 r2 스코어 : 0.6315198682324188
 
 loss :  1939.6295166015625
r2 :  0.7218312395606071
RMSE :  44.04122607897963

같은 파라미터로 돌려도 mean쪽이 더 잘나옴
"""

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv) # train 데이터의 shape와 동일한 컬럼을 확인하고 넣자
                        # x_train.shape: (N, 9)
print(y_submit.shape) # (715, 1)

############## submission.csv 파일 만들기// count 컬럼값만 넣어주기 ####################
submission_csv['count'] = y_submit
# print(submission_csv)

######################## csv파일 만들기 ######################
submission_csv.to_csv(path + 'submission_0522_1102.csv') # csv 만들기.
""""
# 일취월장할것

# loss :  2703.120361328125
# r2 :  0.5818439751886051
# RMSE :  51.991543161076365


# loss :  2523.65185546875
# r2 :  0.5744174867825806
# RMSE :  50.235963154154234


# loss :  2447.112060546875
# r2 :  0.5894286255598356
# RMSE :  49.46829542955814


# loss :  3467.87451171875
# r2 :  0.5202049326579041
# RMSE :  58.88866110480772

############### relu ##########################
# loss :  2080.163330078125
# r2 :  0.6251219926050813
# RMSE :  45.608805271239845

# loss :  2763.540771484375
# r2 :  0.6450211472163702
# RMSE :  52.56938932813437

# loss :  2355.539794921875
# r2 :  0.6974291413620637
# RMSE :  48.533905496646696

# loss :  2055.8154296875
# r2 :  0.7359289575719099
# RMSE :  45.341103036552


# loss :  1314.424560546875
# r2 :  0.8311612043856043
# RMSE :  36.254992036289174

# loss :  1291.823974609375
# r2 :  0.8340642623712313
# RMSE :  35.94195171562843  

# loss :  1138.7061767578125
# r2 :  0.85373235388595
# RMSE :  33.744719890159125 _1613

# loss :  1207.8505859375
# r2 :  0.8448507098745852
# RMSE :  34.75414331906112 _1614

# loss :  774.700927734375
# r2 :  0.900489092967868
# RMSE :  27.833449522851414 _1628

# loss :  791.1190185546875
# r2 :  0.8983802035898768
# RMSE :  28.12683409679516  _1635

# loss :  357.6424560546875
# r2 :  0.9540605577352053
# RMSE :  18.911437877173817 _1704   65점   과적합?

loss :  2522.85498046875
r2 :  0.6178400826013801
RMSE :  50.22803015717718

------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=331
    )

model = Sequential()
model.add(Dense(16, input_dim=9, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs=100, batch_size=32)



------------------------------------------------
 x_train, x_test, y_train, y_test = train_test_split(
     x,y,
     test_size=0.1,
     random_state=231
     )

model = Sequential()
model.add(Dense(600, input_dim=9, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs=500, batch_size=32)

====================================================
새로운 데이터

loss :  2684.410400390625
r2 :  0.6450617739784752
RMSE :  51.81129212943843

loss :  2614.836669921875
r2 :  0.6519681894900944
RMSE :  51.13547340677993

5/22 데이터

loss :  2370.584228515625
r2 :  0.665741425243591
RMSE :  48.68864920641256

loss :  1872.128173828125
r2 :  0.7360250799147048
RMSE :  43.26809562677728

----------------------------------------------
loss :  1489.0689697265625
r2 :  0.7900373918129046
RMSE :  38.58845821163289  _1000 58점
------------------------------------------------

loss :  1129.8006591796875
r2 :  0.8406951916888207
RMSE :  33.61250574862557 _1022   63점

loss :  1132.32421875
r2 :  0.8403393369134453
RMSE :  33.650026582270954 _1102 70점


loss :  1006.0885009765625
r2 :  0.8581389209689267
RMSE :  31.718897008574228  에포400 _1020 과적합  67점대 

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=431
    )

#2. 모델구성


model = Sequential()
model.add(Dense(256, input_dim=9,))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs=300, batch_size=32)   _1569등

"""

