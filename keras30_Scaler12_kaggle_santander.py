# https://www.kaggle.com/competitions/santander-customer-transaction-prediction


import numpy as np                                           # 훈련에 특화됨.
import pandas as pd                                          # 데이터분석을 할 때 전처리, 정제에서 유명함.
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
x = train_csv.drop(['ID','target'], axis=1) # target, ID column만 빼겠다.
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

# scaler=MinMaxScaler()
# scaler=StandardScaler()
# scaler=MaxAbsScaler()
# scaler=RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델구성

model = Sequential()
model.add(Dense(128, input_dim=x.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))  
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))  
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(1, activation='sigmoid'))

path='./_save/keras30/Scaler12_kaggle_santander/'
model.save_weights(path+'keras30_05_kaggle_santander_weights.h5')

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

es = EarlyStopping(                  
    monitor='val_loss',
    mode = 'min',                      
    patience=30,                       
    restore_best_weights=True,        
) 

path='./_save/keras30/Scaler12_kaggle_santander/'
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=path+'keras30_05_kaggle_santander.hdf5'
    )

hist = model.fit(x,y, epochs=300, batch_size=128,
          verbose =2,
          validation_split=0.2,
          callbacks=[es, mcp],
          )

path='./_save/keras30/Scaler12_kaggle_santander/'
model.save(path+'keras30_05_kaggle_santander.h5')

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

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit) # train 데이터의 shape와 동일한 컬럼을 확인하고 넣자
#print(y_submit)                        # x_train.shape: (N, 9)
#print(y_submit.shape) # (715, 1)

############## submission.csv 파일 만들기// count 컬럼값만 넣어주기 ####################
sample_submission_csv['target'] = y_submit
#print(samplesubmission_csv)
# print(submission_csv)

######################## csv파일 만들기 ######################
path='./_save/keras30/Scaler12_kaggle_santander/'
sample_submission_csv.to_csv(path + 'sample_submission_csv_0608_1140.csv', index=False) # csv 만들기.
print("\n📁 'sample_submission_csv' 저장 완료!")

import matplotlib.pyplot as plt        # 맷플로립
import matplotlib.font_manager as fm
import matplotlib as mpl

font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 다른 한글 폰트 경로
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))         # 9X6사이즈로 만들어줘
plt.plot(hist.history['loss'], c='red', label = 'loss')                        # 선그리는게 plot //  loss의 그림을 그리고싶어 // y축은 loss x축은 epochs 훈련량에 따른 loss값 산출  
                                                                               # 리스트는 순서대로 가기때문에 x를따로 명시안해도된다. // y값만 넣으면 시간순으로 그림을 그림
plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')               
plt.title('santander Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  # 우측 상단에 label 표시
plt.grid()                     # 격자표시
plt.show()

"""
loss :  [0.22916652262210846, 0.9146749973297119]
r2 :  0.2684763699467141
RMSE :  0.25715667068133685
"""