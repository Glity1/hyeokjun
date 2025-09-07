### 28-2.copy

# import sklearn as sk
# print(sk.__version__)   # 1.1.3
# import tensorflow as tf
# print(tf.__version__)   # 2.9.3

from tensorflow.python.keras.models import Sequential, load_model
# ✅ 통일된 경로: tensorflow.keras에서만 가져오기
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
# ✅ 권장 방식


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
# print(x)
# print(y)        # 데이터를 찍어보고 소수점(회귀) or 정수 몇개로만 구성(분류) AI 모델 종류 결정
# print(x.shape)  # (20640, 8)
# print(y.shape)  # (20640,)
# print(datasets.feature_names)
#                 # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size= 0.2,
                                              shuffle= True,
                                              random_state= 55)

### Scaling

MS = MinMaxScaler()

MS.fit(x_trn)

x_trn = MS.transform(x_trn)
x_tst = MS.transform(x_tst)



optim = [Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]
lr = [0.1, 0.01, 0.001, 0.0001]  # 학습률을 바꿔가면서 돌려보자]


for opt_class in optim:
    for lr_val in lr:
        #2. 모델구성
        model = Sequential()
        model.add(Dense(128,input_dim=8, activation='relu'))
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
    hist = model.fit(x_trn, y_trn, epochs = 100, batch_size = 32,
            verbose=2,
            validation_split=0.2,
            #   callbacks= [ES,MCP]
            )
    end = time.time()
    # path = './practice/_save/'

    # model.save(path + '20250531_02_california.h5')

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



    # plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows일 경우
    # plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

    # plt.figure(figsize=(9,6)) # 9*6 사이즈 (단위 확인 피룡) # 크기설정이 1번
    # plt.plot(hist.history['loss'], color='red', label='loss')
    # plt.plot(hist.history['val_loss'], color='blue', label='val_loss') # y=epochs로 시간 따라서 자동으로
    # plt.title('캘리포니아 loss') # 표 제목 한글 깨짐해결법 피룡
    # plt.xlabel('epochs') # x 축 이름
    # plt.ylabel('loss') # y 축 이름
    # plt.legend(loc='upper right') # 우측상단 라벨표시 / defualt 는 빈자리
    # plt.grid() # 격자표시


    #4. 평가, 예측
    loss = model.evaluate(x_tst, y_tst)
    results = model.predict([x_tst])

    def RMSE(a, b) :
        return np.sqrt(mean_squared_error(a,b))

    rmse = RMSE(y_tst, results)
    R2 = r2_score(y_tst, results)

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