import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# https://dacon.io/competitions/open/235576/overview/description

import numpy as np                                           # 훈련에 특화됨.
import pandas as pd                                          # 데이터분석을 할 때 전처리 정제에서 유명함.
print(np.__version__)                                        # 1.23.0
print(pd.__version__)                                        # 2.2.3
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


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

input1 = Input(shape=(8,))       # 대문자 클래스 를 인스턴스화 한다.
dense1 = Dense(128, name='layer_1')(input1)       # (input) = input 에서 받아들인다 
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(128, name='layer_2', activation='relu')(drop1)        # name='원하는 이름 ' : summary에서 보이는 레이어의 이름을 지을 수 있음
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(128, name='layer_3', activation='relu')(drop2)         # 임의적으로 순서를 바꿔서 모델을 구성 가능하다.
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(128, name='layer_4', activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(64, name='layer_5', activation='relu')(drop4)
drop5 = Dropout(0.1)(dense5)
dense6 = Dense(64, name='layer_6', activation='relu')(drop5)
drop6 = Dropout(0.1)(dense6)
dense7 = Dense(64, name='layer_7', activation='relu')(drop6)
drop7 = Dropout(0.1)(dense7)
output1 = Dense(1)(drop7)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping  #callback : 불러와
es = EarlyStopping(                   # EarlyStopping을 es라는 변수에 넣는다
    monitor='val_loss',
    mode = 'min',                      # 최대값 max, 알아서 찾아줘 : auto /통상 min 이 default
    patience=30,                      # patience이 작으면 지역최소에 빠질수있음.  (history상에 10번 참는다는것은 마지막값에서 11번째 값이 최소값으로 보여준다.)
    restore_best_weights=True,        # 가장 최소 지점으로 저장한다
) 

path='./_save/keras28_mcp/02_california/'
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=path+'keras35_mcp_02_california.hdf5'
    )

hist = model.fit(x_train,y_train, epochs=200, batch_size=64,
          verbose =1,
          validation_split=0.2,
          callbacks=[es, mcp],
          ) 

# path='./_save/keras28_mcp/02_california/'
# model.save(path+'keras28_mcp_save_02_california.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

print("loss : ", loss)
print("[x_test]의 예측값 : ", results)

r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)

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
plt.title('켈리포니아 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  # 우측 상단에 label 표시
plt.grid()                     # 격자표시
plt.show()

"""
#EarlyStopping 적용 후
    patience=30,     
) 

hist = model.fit(x_train,y_train, epochs=10000, batch_size=64,
          verbose =1,
          validation_split=0.2,
          callbacks=[es],
          val_loss = 0.6309
          
loss :  0.6327128410339355 
r2 스코어 :  0.5288090954956464

robustscaler 적용 후
loss :  0.5183307528495789
r2 스코어 :  0.6139912582344228
       
dropout 적용 후
          
"""