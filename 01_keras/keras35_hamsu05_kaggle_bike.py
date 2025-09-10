# https://www.kaggle.com/competitions/bike-sharing-demand/leaderboard?

import numpy as np                                           # 훈련에 특화됨.
import pandas as pd                                          # 데이터분석을 할 때 전처리, 정제에서 유명함.
print(np.__version__)                                        # 1.23.0
print(pd.__version__)                                        # 2.2.3
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
path = './_data/kaggle/bike/'      # 역슬래시 예약되있는 문자 kaggle\b 까지 폴더로봄     //////  상대경로
#path = '.\_data\kaggle\\bike\\'        # \n, \b 등 노란색 밑줄 잘볼것  섞어쓰면 가독성이 떨어짐. ㅈ\
#path = '.\\_data\\kaggle\\bike\\'
#path = './/_data//kaggle//bike//'

### 절대경로 ###
#path = 'c:/Study25/_data/kaggle/bike/' #경로 처음부터 끝까지 다 써주는것 /// 절대경로


train_csv = pd.read_csv(path + 'train.csv', index_col=0) # datetime column을 index 로 변경 아래 세줄 동일
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
samplesubmission_csv = pd.read_csv(path + 'samplesubmission.csv')
print(samplesubmission_csv) #[6493 rows x 2 columns]
# exit()
print(train_csv)                        # [10886 rows x 11 columns]
print(train_csv.columns)                # Index(['season', 'holiday', 'workingday', 'weather', 'temp',
                                        # 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
                                        # dtype='object')                            
print(test_csv)                         # [6493 rows x 8 columns]
print(test_csv.columns)                 # Index(['season', 'holiday', 'workingday', 'weather', 'temp',
                                        #  'atemp', 'humidity', 'windspeed'],
                                        #  dtype='object')                        
print(samplesubmission_csv)             # [6493 rows x 2 columns]
print(samplesubmission_csv.columns)     # Index(['datetime','count'], dtype='object') 


# 결측치 확인
    
print(train_csv.info())
print(train_csv.isnull().sum())         #결측치 없음
print(test_csv.isna().sum())            #결측치 없음

print(train_csv.describe())

# #################### x,y 데이터를 분리한다 ##########################
print('###############################################################################')
x = train_csv.drop(['count', 'casual', 'registered'], axis=1) # axis =1 컬럼 // count, casual, registered 삭제
print(x) #[10886 rows x 8 columns]
                       
y = train_csv['count']               # count column만 빼서 y에 넣겠다.
print(y)  
print(y.shape)                       # (10886,)



x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=74, # validation_split로도 바뀌지않는다면 바꾸자
    )

# #2. 모델구성

# model = Sequential()
# model.add(Dense(256, input_dim=8)) #  
# model.add(Dropout(0.3))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='linear')) # linear => y=wx+b 이게 default 표기를 따로 안하면 linear 들어가있음.
                                         # 끝에 linear를 넣어주면 좋았었다.

# path='./_save/keras28_mcp/05_kaggle_bike/'
# model.save_weights(path+'keras33_MCP_save_05_kaggle_bike.h5')

input1 = Input(shape=(8,))       # 대문자 클래스 를 인스턴스화 한다.
dense1 = Dense(256, name='layer_1')(input1)       # (input) = input 에서 받아들인다 
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(256, name='layer_2', activation='relu')(drop1)        # name='원하는 이름 ' : summary에서 보이는 레이어의 이름을 지을 수 있음
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(256, name='layer_3', activation='relu')(drop2)         # 임의적으로 순서를 바꿔서 모델을 구성 가능하다.
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(256, name='layer_4', activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(128, name='layer_5', activation='relu')(drop4)
drop5 = Dropout(0.2)(dense5)
dense6 = Dense(128, name='layer_6', activation='relu')(drop5)
drop6 = Dropout(0.2)(dense6)
dense7 = Dense(128, name='layer_7', activation='relu')(drop6)
drop7 = Dropout(0.2)(dense7)
dense8 = Dense(128, name='layer_7', activation='relu')(drop7)
drop8 = Dropout(0.1)(dense8)
dense9 = Dense(64, name='layer_8', activation='relu')(drop8)
drop9 = Dropout(0.1)(dense9)
dense10 = Dense(64, name='layer_9', activation='relu')(drop9)
drop10 = Dropout(0.1)(dense10)
dense11 = Dense(64, name='layer_10', activation='relu')(drop10)
drop11 = Dropout(0.1)(dense11)
dense12 = Dense(64, name='layer_11', activation='relu')(drop11)
drop12 = Dropout(0.1)(dense12)
output1 = Dense(1)(drop12)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping  #callback : 불러와
es = EarlyStopping(                   # EarlyStopping을 es라는 변수에 넣는다
    monitor='val_loss',
    mode = 'min',                      # 최대값 max, 알아서 찾아줘 : auto /통상 min 이 default
    patience=30,                      # patience이 작으면 지역최소에 빠질수있음.  (history상에 10번 참는다는것은 마지막값에서 11번째 값이 최소값으로 보여준다.)
    restore_best_weights=True,        # 가장 최소 지점으로 저장한다
) 

path='./_save/keras28_mcp/05_kaggle_bike/'
mcp=ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,    
    filepath=path+'keras35_mcp_05_kaggle_bike.hdf5'
    )

hist = model.fit(x,y, epochs=300, batch_size=64,
          verbose =1,
          validation_split=0.2,
          callbacks=[es, mcp],
          )

path='./_save/keras28_mcp/05_kaggle_bike/'
model.save(path+'keras35_mcp_save_05_kaggle_bike.h5')

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
#print(y_submit)                        # x_train.shape: (N, 9)
#print(y_submit.shape) # (715, 1)

############## submission.csv 파일 만들기// count 컬럼값만 넣어주기 ####################
samplesubmission_csv['count'] = y_submit
#print(samplesubmission_csv)
# print(submission_csv)

######################## csv파일 만들기 ######################
samplesubmission_csv.to_csv(path + 'samplesubmission_0527_1050.csv', index=False) # csv 만들기.

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
plt.title('캐글 자전거 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')  # 우측 상단에 label 표시
plt.grid()                     # 격자표시
plt.show()

"""
loss :  25247.287109375
r2 :  0.21763124622477859
RMSE :  158.8939429925698
 
dropout 적용 후
loss :  26472.01171875
r2 :  0.17967929701738627
RMSE :  162.70220166451867
          
"""          


