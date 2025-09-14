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
# ✅ 권장 방식


#1. 데이터
path = './_data/dacon/diabetes/'      

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   
print(train_csv)                    # [652 rows x 9 columns]             

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(test_csv)                     # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(submission_csv)               # [116 rows x 1 columns] 

print(train_csv.shape)              #(652,9)
print(test_csv.shape)               #(116,8)
print(submission_csv.shape)         #(116,1)

print("######################################################################################################################")                                              
print(train_csv.columns)  # Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          # 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
                          # dtype='object')
print("######################################################################################################################")                                              
print(train_csv.info())             
print("######################################################################################################################")                          
print(train_csv.describe())

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome'] 
x = x.replace(0, np.nan)                
x = x.fillna(x.mean())
test_csv = test_csv.fillna(x.mean())

print(train_csv.isna().sum())
print(test_csv.isna().sum())

print(x) #652 rows x 8 
print(y) #652

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=813, shuffle=True,
    )
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