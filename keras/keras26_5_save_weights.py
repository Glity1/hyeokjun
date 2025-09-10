#26-3 copy

import sklearn as sk
print(sk.__version__)   # 1.1.3
from sklearn.datasets import load_boston
import time
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping

# 1.1 데이터 로드
datasets= load_boston()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
                              # 'B' 'LSTAT']

x=datasets.data
y=datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test=train_test_split(
    x,y, train_size=0.8, shuffle=True,
    random_state=333,
)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))   # -0.00557837618540494 1.1478180091225068

# 2 모델 구성
model=Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(1))

model.summary() # Total params: 588

path='./_save/keras26/'
# model.save(path+'keras26_1_save.h5')
model.save_weights(path+'keras26_5_save1.h5')

# 3 컴파일, 훈련 (19_1  copy)
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(# EarlyStopping :모델이 최적의 성능을 보이는 시점에서 훈련을 멈추도록 돕는 기법
    monitor='val_loss',
    mode = 'min', # 최대값 : max(accuary 측정할 때), 알아서 찾아줘 : auto
    patience=50, # patience=10이면 최소값 이후 10번 더 참았다가 성능이 더 이상 개선되지 않으면 종료
    restore_best_weights=True,  # 가장 좋은 가중치의 결과, default=True
)


hist = model.fit(x_train,y_train, epochs=100, batch_size=32, # loss, val_loss의 epochs의 수만큼 값을 반환해서 넣어준다 // 리스트 값이된다 2개니까
          verbose=3,                            # return 이 포함된다.
          validation_split=0.2,
          callbacks=[es],
          )

path='./_save/keras26/'
model.save(path+'keras26_5_save2.h5')

#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict([x_test])

print("loss : ", loss)
print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)
# r2 기준으로 0.75이상
# loss :   29.726648330688477
# r2 스코어 : 0.7584895173069439
# r2 스코어 :  0.8484866157781902 로 상승