import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, mean_squared_error
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))

print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# dtype: int64

# y = to_categorical(y)
y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)
print(y.shape) #(581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=814,
    shuffle=True, stratify=y # 0,1,2 를 균등하게
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# model = Sequential()
# model.add(Dense(128, input_dim=x.shape[1], activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(7, activation='softmax'))

path='./_save/keras28_mcp/10_fetch_covtype/'
# 체크포인트로 확인
model=load_model(path+'keras28_mcp_10_fetch_covtype.hdf5') 
# 세이브 모델 확인
model=load_model(path+'keras28_mcp_save_10_fetch_covtype.h5')

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
# es = EarlyStopping(monitor='val_loss', 
#                    mode='min', 
#                    patience=30, 
#                    restore_best_weights=True)

# path='./_save/keras28_mcp/10_fetch_covtype/'
# mcp=ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,    
#     filepath=path+'keras28_mcp_10_fetch_covtype.hdf5'
#     )

# hist = model.fit(
#     x_train, y_train,
#     epochs=200,
#     batch_size=1024,
#     validation_split=0.2,
#     callbacks=[es, mcp],
#     verbose=2
# )

# path='./_save/keras28_mcp/10_fetch_covtype/'
# model.save(path+'keras28_mcp_save_10_fetch_covtype.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, results)

print("loss : ", loss)
print("RMSE : ", rmse)
print("r2 스코어 : ", r2)

"""
loss :  [0.2843846082687378, 0.8849771618843079]
RMSE :  0.15484191435992453
r2 스코어 :  0.7061185085888007

"""
