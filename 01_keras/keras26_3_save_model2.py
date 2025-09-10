import sklearn as sk
print(sk.__version__)
from sklearn.datasets import load_boston
import  numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

#1. 데이터
datasets = load_boston()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']

x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=5176
)

# 스케일링 split 후에 해야한다. 전체 성능상 버릴놈은 버리고 시작하는게 맞다.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()             # 변수//인스턴스
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train)) # min  0.0                  max  1.0000000000000002
print(np.min(x_test), np.max(x_test))   # min -3.0797887849379e-05  max  1.0280851063829786

# a = 0.1
# b = 0.2
# print(a+b) # 0.30000000000000004 부동소수점연산할때 오차가 생긴다. 그냥 0.3이라고 보자

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(1))

# path = './_save/keras26/'
# model.save(path + 'keras26_1_save.h5')

#3. 컴파일, 훈련 (19-1 copy)
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping  
es = EarlyStopping(                   # EarlyStopping을 es라는 변수에 넣는다
    monitor='val_loss',
    mode = 'min',                      # 최대값 max, 알아서 찾아줘 : auto /통상 min 이 default
    patience=20,                      # patience이 작으면 지역최소에 빠질수있음.  (history상에 10번 참는다는것은 마지막값에서 11번째 값이 최소값으로 보여준다.)
    restore_best_weights=True,        # 가장 최소 지점으로 저장한다
) 

model.fit(x,y, epochs=100, batch_size=16,      # loss, val_loss의 epochs의 수만큼 값을 반환해서 넣어준다 // 리스트 값이된다 2개니까
          verbose=1,                                  # return 이 포함된다.
          validation_split=0.2,
          callbacks=[es], 
          )       

path = './_save/keras26/'
model.save(path + 'keras26_3_save.h5')

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

# loss :  534.4879150390625
# r2 :  -6.378842784979296
# RMSE :  23.11899518029537