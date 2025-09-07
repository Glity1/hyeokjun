# https://www.kaggle.com/competitions/santander-customer-transaction-prediction


import numpy as np                                           # 훈련에 특화됨.
import pandas as pd                                          # 데이터분석을 할 때 전처리, 정제에서 유명함.
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
path = './_data/kaggle/santander/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')

# print(sample_submission_csv) #[200000 rows x 2 columns]

# print(train_csv)                        # [200000 rows x 201 columns]
# print(train_csv.columns)                # Index(['target', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6',
#                                         #    'var_7', 'var_8',
#                                         #    ...
#                                         #    'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
#                                         #    'var_196', 'var_197', 'var_198', 'var_199'],
#                                         #   dtype='object', length=201)
#                                         # dtype='object')        
                                                             
# print(test_csv)                         # [200000 rows x 200 columns]
# print(test_csv.columns)                 # Index(['var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7',
#                                         #        'var_8', 'var_9',
#                                         #        ...
#                                         #        'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
#                                         #        'var_196', 'var_197', 'var_198', 'var_199'],
#                                         # dtype='object', length=200)                       
# print(sample_submission_csv)             # [200000 rows x 2 columns]
# print(sample_submission_csv.columns)     # Index(['datetime','count'], dtype='object') 


# 결측치 확인
    
# print(train_csv.info())                 
# print(train_csv.isnull().sum())         #결측치 없음
# print(test_csv.isna().sum())            #결측치 없음

# print(train_csv.describe())             


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

print(x_train.shape, x_test.shape)  # (160000, 200) (40000, 200)
print(y_train.shape, y_test.shape)  # (160000,) (40000,)

scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

# exit()

x_train = x_train.reshape(160000,25,8,1)
x_test = x_test.reshape(40000,25,8,1)

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, padding='same', input_shape=(25, 8, 1)))
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Activation('relu')) 
model.add(Dropout(0.1))
model.add(Dense(units=1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(                  
    monitor='val_loss',
    mode = 'min',                      
    patience=30,                       
    restore_best_weights=True,        
) 

hist = model.fit(x_train,y_train, epochs=300, batch_size=128,
          verbose =2,
          validation_split=0.2,
          callbacks=[es],
          )


print('========================hist============================')
print(hist) # keras.callbacks.History object at 0x000002AE4E644220> 으로 나오는데 제대로 볼려면 
print('========================hist.history============================')
print(hist.history)  # 중괄호의 등장 : 키(loss, val_loss) : 벨류(숫자) 형태로 안에 넣어둔다 // loss, val loss 의 갯수는 epochs 값과 똑같음
                     # loss들의 역사 
                     # 그래프의 시각화가 가능하다 점들의 값이 있기 떄문에

print('========================hist.history에서 loss만 따로보고싶다============================')
print(hist.history['loss'])   # dictionary의 키값만 적어주면된다                     
       
print('========================hist.history에서 val_loss만 따로보고싶다============================')
print(hist.history['val_loss'])   # dictionary의 키값만 적어주면된다

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict(x_test)

best_thresh = 0.5
best_f1 = 0

for thresh in np.arange(0.1, 0.9, 0.01):
    pred_bin = (results > thresh).astype(int).reshape(-1)
    f1 = f1_score(y_test, pred_bin)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Best threshold: {best_thresh}, Best F1 Score: {best_f1:.4f}")
print("loss : ", loss)

