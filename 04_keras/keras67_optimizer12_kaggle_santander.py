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
import pandas as pd  
# ✅ 권장 방식


#1. 데이터
path = './_data/kaggle/santander/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(sample_submission_csv) #[200000 rows x 2 columns]

print(train_csv)                        # [200000 rows x 201 columns]
print(train_csv.columns)                # Index(['target', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6',
                                        #    'var_7', 'var_8',
                                        #    ...
                                        #    'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
                                        #    'var_196', 'var_197', 'var_198', 'var_199'],
                                        #   dtype='object', length=201)
                                        # dtype='object')        
                                                             
print(test_csv)                         # [200000 rows x 200 columns]
print(test_csv.columns)                 # Index(['var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7',
                                        #        'var_8', 'var_9',
                                        #        ...
                                        #        'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
                                        #        'var_196', 'var_197', 'var_198', 'var_199'],
                                        # dtype='object', length=200)                       
print(sample_submission_csv)             # [200000 rows x 2 columns]
print(sample_submission_csv.columns)     # Index(['datetime','count'], dtype='object') 


# 결측치 확인
    
print(train_csv.info())                 
print(train_csv.isnull().sum())         #결측치 없음
print(test_csv.isna().sum())            #결측치 없음

print(train_csv.describe())             


# #################### x,y 데이터를 분리한다 ##########################
print('###############################################################################')
x = train_csv.drop(['target'], axis=1) # target, ID column만 빼겠다.
print(x)                               # [200000 rows x 200 columns]
                       
y = train_csv['target']                # target column만 빼서 y에 넣겠다.
print(y)  
print(y.shape)                         # (200000,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=74, # validation_split로도 바뀌지않는다면 바꾸자
    stratify=y
    )

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
        model.add(Dense(128,input_dim=200, activation='relu'))
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