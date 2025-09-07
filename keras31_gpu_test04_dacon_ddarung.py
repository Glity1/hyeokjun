# https://dacon.io/competitions/open/235576/overview/description
import tensorflow as tf
import time
import numpy as np                                           # 훈련에 특화됨.
import pandas as pd                                          # 데이터분석을 할 때 전처리 정제에서 유명함.
print(np.__version__)                                        # 1.23.0
print(pd.__version__)                                        # 2.2.3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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

path='./_save/keras28_mcp/04_dacon_ddarung/'
model.save_weights(path+'keras28_MCP_save_04_dacon_ddarung.h5')

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

start = time.time()
hist = model.fit(x,y, epochs=100, batch_size=32,
          verbose =1,
          validation_split=0.2,
        #   callbacks=[es, mcp],
          )
end = time.time()

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')
print("걸린시간 : ", round(end - start), "초")

# GPU 있다~
# 걸린시간 : 18초

# GPU 없다~
# 걸린시간 :
"""
path='./_save/keras28_mcp/04_dacon_ddarung/'
model.save(path+'keras28_mcp_save_04_dacon_ddarung.h5')

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

print("loss : ", loss)
r2 = r2_score(y_test, results)
print('r2 : ', r2)

def RMSE(y_test, results):
    return np.sqrt(mean_squared_error(y_test, results)) 
rmse = RMSE(y_test, results) 
print('RMSE : ', rmse)

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv) # train 데이터의 shape와 동일한 컬럼을 확인하고 넣자
                        # x_train.shape: (N, 9)
print(y_submit.shape) # (715, 1)

############## submission.csv 파일 만들기// count 컬럼값만 넣어주기 ####################
submission_csv['count'] = y_submit
# print(submission_csv)

######################## csv파일 만들기 ######################
# submission_csv.to_csv(path + 'submission_0526_1503.csv') # csv 만들기.

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
plt.title('따릉이 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  # 우측 상단에 label 표시
plt.grid()                     # 격자표시
plt.show()


loss :  2671.0166015625
r2 :  0.6535899506298574
RMSE :  51.68188076545843    
                    
"""          