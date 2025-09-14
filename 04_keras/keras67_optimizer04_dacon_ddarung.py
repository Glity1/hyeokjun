from keras.models import Sequential, load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    random_state=53 # validation_split로도 바뀌지않는다면 바꾸자
    )
### Scaling

MS = MinMaxScaler()

MS.fit(x_train)

x_train = MS.transform(x_train)
x_test = MS.transform(x_test)


optim = [Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]
lr = [0.1, 0.01, 0.001, 0.0001]  # 학습률을 바꿔가면서 돌려보자]


for opt_class in optim:
    for lr_val in lr:
        #2. 모델구성
        model = Sequential()
        model.add(Dense(128,input_dim=9, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(BatchNormalization())
        model.add(Dense(1))
        
        epochs = 100
        
        #3. 컴파일, 훈련
        optimizer = opt_class(learning_rate=lr_val)
        model.compile(loss='mse', optimizer=optimizer)





    ES = EarlyStopping(monitor='val_loss',
                    mode= 'min',
                    patience= 100,
                    restore_best_weights= True)

    ################################# mpc 세이브 파일명 만들기 #################################
    ### 월일시 넣기
    import datetime

    path_MCP = './_save/keras28_mcp/02_california/'

    date = datetime.datetime.now()
    # print(date)            
    # print(type(date))       
    date = date.strftime('%m%d_%H%M')              

    # print(date)             
    # print(type(date))

    filename = '{epoch:04d}-{val_loss:.4f}.h5'
    filepath = "".join([path_MCP,'keras28_',date, '_', filename])

    MCP = ModelCheckpoint(monitor='val_loss',
                        mode='auto',
                        save_best_only=True,
                        filepath= filepath # 확장자의 경우 h5랑 같음
                                            # patience 만큼 지나기전 최저 갱신 지점        
                        )
    import time
    start = time.time()
    hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32,
            verbose=2,
            validation_split=0.2,
            #   callbacks= [ES,MCP]
            )
    end = time.time()

    ''' 20250531 갱신
    loss : 0.28095752000808716
    rmse : 0.5300542568053527
    R2 : 0.792117619711086

    [MCP]
    loss : 0.2854580879211426
    rmse : 0.5342827826116628
    R2 : 0.788787612383781

    [load]

    '''
    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test)
    results = model.predict([x_test])

    def RMSE(a, b) :
        return np.sqrt(mean_squared_error(a,b))

    rmse = RMSE(y_test, results)
    R2 = r2_score(y_test, results)

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)

    if gpus:
        print('GPU 있다!!!')
    else:
        print('GPU 없다...')
        
    time = end - start
    print("소요시간 :", time)


    '''
    GPU 있다!!!
    소요시간 : 279.6294457912445

    GPU 없다...
    소요시간 : 45.43129324913025

    '''