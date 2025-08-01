import numpy as np                                           # 훈련에 특화됨.
import pandas as pd                                          # 데이터분석을 할 때 전처리, 정제에서 유명함.
print(np.__version__)                                        # 1.23.0
print(pd.__version__)                                        # 2.2.3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

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
# x = train_csv.drop(['count', 'casual', 'registered'], axis=1) # axis =1 컬럼 // count, casual, registered 삭제
x = train_csv.drop(['casual', 'registered','count'], axis=1) # axis =1 컬럼 // count, casual, registered 삭제
print(x) #[10886 rows x 8 columns]
                       
y = train_csv[['casual','registered']]                  # count column만 빼서 y에 넣겠다.
print(y)  
print(y.shape)                          # (10886,2)

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,          #열 위주로 잘라준다 가령 (10행 3열 ) = (8행 3열) + (2행 3열) 열은 건들지않고 행을 자르고 열이 기준이 된다.
    random_state=74,
    )

# #2. 모델구성

model = Sequential()
model.add(Dense(128, input_dim=8)) #  
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(2, activation='linear'))  # output layer 로드의 갯수를 바꿔야한다 1->2
                                         
#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs=10, batch_size=32)

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

############ 두개의 타입이 다른 데이터를 같이 사용할때 #############

print('test.csv 타입 : ', type(test_csv))
# <class 'pandas.core.frame.DataFrame'>
# 시리즈(series): 벡터 데이터프레임(dataframe): 행렬

print('y_submit 타입 : ', type(y_submit))
# <class 'numpy.ndarray'>
# 두 데이터의 타입이 다른데 문제가 없느가? #
# 결론 pandas 안에 데이터가 numpy 와 같이 사용가능하다.

exit()
################# 원본을 건들지 않기위해 복사 파일을 사용한 작업#############

test2_csv = test_csv                    # 원래는 .copy()를 사용해야함. a=b b를 a에다가 넣겠다.


############## submission.csv 파일 만들기// count 컬럼값만 넣어주기 ####################
test2_csv[['casual', 'registered']] = y_submit   # test_csv의 사본을 만들어서 작업하는게 맞다 (지금 작업은 원본에다가 casual과 registered를 넣기 때문에 원본을 훼손시킨다.)
print(test2_csv)

# print(samplesubmission_csv)
# print(test2_csv)


############################# csv파일 만들기 ###############################
test2_csv.to_csv(path + 'new_test.csv') # csv 만들기.